from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.patches as patches
from matplotlib.patches import Ellipse


from pathlib import Path
from loc_tools import crb_minflux
from configvar import (
    NUM_PULSES,
    STEP_NM,
    SIGMA_TOL_OUTLIERS,
)
from ebp import EBP

class DataPostProcessor():
    def __init__(self, locs_filepath: Path, drift_data_filepath: Path, t_start_filepath: Path, ebp: EBP, bin_size: int, use_drift_data_choice: bool):
        self.locs_filepath = locs_filepath
        self.drift_data_filepath = drift_data_filepath
        self.t_start_filepath = t_start_filepath
        self.ebp = ebp
        self.bin_size = bin_size
        self.use_drift_data_choice = use_drift_data_choice
        
        # open file containing localization results
        self.locs = self.load_locs(self.locs_filepath)
        if self.use_drift_data_choice:
            # open file with drift data for a posteriori drift correction
            self.drift_data = self.prepare_drift_data(self.drift_data_filepath)
            # correct a posteriori localizations with drift data
            self.locs = self.apost_drift_locs_corr(self.locs, self.drift_data)
        # eliminate spatial outliers from localizations
        self.locs_nooutliers = self.locs #self.eliminate_outliers(self.locs)
        # now filter localizations based on photons numbers
        self.locs_filt = self.filter_locs_forph(self.locs_nooutliers)
        # compute average photon number and SBR based on filtered localizations
        self.avg_ph_perloc, self.avg_sbr = self.get_avg_loc_param(self.locs_filt)
        # compute and plot CRB map, superimposed with EBP
        self.calc_crb(self.avg_ph_perloc, self.avg_sbr)
        self.plot_crb_andebp()
        # recenter localizations
        self.locs_centered = self.center_locs(self.locs_filt)
        # get range for future plots
        self.x_plot_range, self.y_plot_range = self.get_glob_plots_limits(self.locs_centered)
        # all plots
        self.plot_locs_timecoded_withebp(self.locs_centered)
        self.plot_loc_density_withebp(self.locs_centered)
        self.plot_locs_withcrb(self.locs_centered)
        # self.plot_locs_gmm_filtered(self.locs_centered)
        
    def load_locs(self, locs_filepath):
        """
        This function loads the localization array saved as a .npy
        """
        locs = np.load(locs_filepath)
        self.avg_ph_perloc = np.mean(locs[:,3])
        self.sigma_ph_perloc = np.std(locs[:,3])
        print(f"Average number of photons per localization: {self.avg_ph_perloc} \u00B1 {self.sigma_ph_perloc}")
        return locs
        
    def load_takyaq_data(self, filename):
        """
        This function reads the output data of the takyaq stabilization system
        """
        with open(filename, 'rb') as fd:
            data = []
            n_batches = 0
            try:
                while True:
                    data.append(np.load(fd))
                    n_batches += 1
            except EOFError:
                print(f"Loaded {n_batches} batches of lengths {[len(x) for x in data]} data points")
        rv = np.concatenate(data) if data else None
        return rv
        
    def prepare_drift_data(self, drift_data_filepath):
        """
        This function loads and prepare drift data for a posteriori correction
        """
        # Load the .npy files into numpy arrays
        xyvst = self.load_takyaq_data(drift_data_filepath)

        # Extract time data
        t = xyvst['t']
        
        # get absolute time of start TCSPC measurement to correct offset
        with open(self.t_start_filepath) as ff:
            t_tcspc_start_s = float(ff.read())
        t_drift_data_start_s = t[0]
        
        # set correct offset in time
        t = t - t[0] - t_tcspc_start_s + t_drift_data_start_s

        # Automatically detect the number of columns in xyvst['xy']
        n_columns = xyvst['xy'].shape[1]  # Number of columns

        # Initialize lists to store x, y data
        x_data = []
        y_data = []

        # Extract x and y columns
        for i in range(n_columns):
            x_data.append(xyvst['xy'][:, i][:, 0])  # Extract x_i
            y_data.append(xyvst['xy'][:, i][:, 1])  # Extract y_i

        # Convert lists to numpy arrays for easier manipulation
        x_data = np.array(x_data).T  # Transpose to match shape (n_samples, n_columns)
        y_data = np.array(y_data).T  # Transpose to match shape (n_samples, n_columns)

        # Compute the average of x and y columns
        x_avg = np.mean(x_data, axis=1)  # Average across columns (axis=1)
        y_avg = np.mean(y_data, axis=1)  # Average across columns (axis=1)

        return np.stack((t, x_avg, y_avg))
        
    def apost_drift_locs_corr(self, locs, drift_data):
        """
        This function uses the data of the xy drift to correct a posteriori the MINFLUX localizations. For each localization
        it uses the closest (in time) datapoint of the drift
        """
        last_closest_t_idx = 0
        for loc_idx in range(len(locs[:, 0])):
            for t_drift_idx in range(last_closest_t_idx, len(drift_data[0])):
                if drift_data[0][t_drift_idx] > locs[loc_idx, 0]:
                    # to start search from here for next point
                    last_closest_t_idx = t_drift_idx - 1
                    locs[loc_idx, 1] -= drift_data[1][t_drift_idx - 1]
                    locs[loc_idx, 2] -= drift_data[2][t_drift_idx - 1]
                    break
        self.locs_driftcorr_results_filename = self.locs_filepath.stem + '_driftcorr_'  + '.npy'
        self.locs_driftcorr_results_filepath = self.locs_filepath.parent / self.locs_driftcorr_results_filename
        np.save(self.locs_driftcorr_results_filepath, self.locs)
        return locs
        
    def eliminate_outliers(self, locs):
        """
        This function eliminates all localization exceeding 3 sigma from the center of mass
        """
        self.average_coords_locs = (
            np.mean(locs[:, 1]),
            np.mean(locs[:, 2])
        )
        self.average_sigma_locs = (
            np.std(locs[:, 1]),
            np.std(locs[:, 2])
        )
        dists_loc_from_center = np.sqrt(
            (locs[:, 1] - self.average_coords_locs[0])**2 + (locs[:, 2] - self.average_coords_locs[1])**2
        )
        locs_nooutliers = locs[dists_loc_from_center < SIGMA_TOL_OUTLIERS * np.sqrt(self.average_sigma_locs[0]**2 + self.average_sigma_locs[1]**2)]
        return locs_nooutliers
        
    def filter_locs_forph(self, locs):
        """This function filters out localizations obtained with less photons than a chosen threshold"""
        min_ph_perloc_input = input("Minimum number of photons required for a single localization (no input for no filtering): ")
        if min_ph_perloc_input:
            self.min_ph_perloc = float(min_ph_perloc_input)
        else:
            self.min_ph_perloc = 0
        locs_filtered = locs[locs[:, 3] >= self.min_ph_perloc]
        return locs_filtered
        
    def get_avg_loc_param(self, locs):
        """
        This function computes the average number of photons and SBR for the remaining localizations
        """
        avg_ph_perloc = np.mean(locs[:, 3])
        avg_sbr = np.average(locs[:, 4], weights=locs[:, 3])
        return avg_ph_perloc, avg_sbr
        
    def center_locs(self, locs):
        """
        This function changes system of reference for the localizations, cenetering their values in the minimum of the first beam
        """
        locs_centered = deepcopy(locs)
        locs_centered[:, 1] -= self.ebp.pos_mins_nm[0][0]
        locs_centered[:, 2] -= self.ebp.pos_mins_nm[0][1]
        return locs_centered
        
    def get_glob_plots_limits(self, locs):
        """
        This function computes the limit in x and y for plots to keep the size of the plotted area consistent
        """
        x_ebp_range = (
            min(self.ebp.pos_mins_centered_nm, key=lambda elem: elem[0])[0],
            max(self.ebp.pos_mins_centered_nm, key=lambda elem: elem[0])[0]
        )
        y_ebp_range = (
            min(self.ebp.pos_mins_centered_nm, key=lambda elem: elem[1])[1],
            max(self.ebp.pos_mins_centered_nm, key=lambda elem: elem[1])[1]
        )
        x_filt_locs_range = (np.min(locs[:, 1]), np.max(locs[:, 1]))
        y_filt_locs_range = (np.min(locs[:, 2]), np.max(locs[:, 2]))
        x_global_range = (np.min((x_ebp_range[0], x_filt_locs_range[0])), np.max((x_ebp_range[1], x_filt_locs_range[1])))
        y_global_range = (np.min((y_ebp_range[0], y_filt_locs_range[0])), np.max((y_ebp_range[1], y_filt_locs_range[1])))
        # add a 10% padding
        x_plot_range = (x_global_range[0] - 0.1 * (x_global_range[1] - x_global_range[0]), x_global_range[1] + 0.1 * (x_global_range[1] - x_global_range[0]))
        y_plot_range = (y_global_range[0] - 0.1 * (y_global_range[1] - y_global_range[0]), y_global_range[1] + 0.1 * (y_global_range[1] - y_global_range[0]))
        return x_plot_range, y_plot_range
                    
    def calc_crb(self, ph_perloc, sbr):
        """
        This function computes the crb based on the experimental SBR and average photon number (averaging x and y errors)
        """
        print(f"Computing CRB with {ph_perloc} photons and {sbr} of SBR")
        self.σ_CRB = crb_minflux(NUM_PULSES, self.ebp.psf_fits, sbr, STEP_NM, self.ebp.size_nm, ph_perloc, method='1')
     
    def plot_crb_andebp(self):
        """
        This function plots the CRB map alone
        """
        # Create the CRB plot with the same extent as the scatter plots
        plt.figure('CRB_map')
        # Shift CRB map by the same amount as the EBP
        plt.imshow(
            self.σ_CRB, cmap='viridis', vmin=0, vmax=20,
            extent=(
                - self.ebp.pos_mins_nm[0][0] - 0.5, self.σ_CRB.shape[1] - self.ebp.pos_mins_nm[0][0] - 0.5,
                - self.ebp.pos_mins_nm[0][1] - 0.5, self.σ_CRB.shape[0] - self.ebp.pos_mins_nm[0][1] - 0.5
            ),
            origin='lower'
        )
        plt.colorbar(label='σ_CRB Value')

        # Plot PSF minima positions with the same color mapping as before
        for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
            plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)

        # Ensure the axes and aspect ratio are the same as in scatter plots
        plt.gca().set_aspect('equal')
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.title('σ_CRB with Aligned Reference Frame')
        plt.tight_layout()
        plt.show()
                        
    def plot_locs_timecoded_withebp(self, locs):
        """
        This function plots all (filtered) localizations, encoding with time, superposed with the EBP
        """
        plt.figure('Time-encoded localizations')
        for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
            plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)
        plt.scatter(locs[:, 1], locs[:, 2], c=(locs[:, 0] - locs[0, 0]), cmap='rainbow', s=20, alpha=0.05)
        color_bar = plt.colorbar(label="Time [s]", orientation="vertical")
        color_bar.solids.set(alpha=1)
        plt.xlim(self.x_plot_range)
        plt.ylim(self.y_plot_range)
        plt.xlabel('x (nm)', fontsize = 14)
        plt.ylabel('y (nm)', fontsize = 14)
        
        # Annotations
        plt.gca().set_aspect('equal'), plt.xlabel('x (nm)'), plt.ylabel('y (nm)'), plt.tight_layout()
        plt.show()
        
    def plot_loc_density_withebp(self, locs):
        """
        This function produces the 2D histogram of the localization density, superposed with the EBP
        """
        bin_x_edges = np.arange(int(self.x_plot_range[0]), int(self.x_plot_range[1]) + self.bin_size, self.bin_size)
        bin_y_edges = np.arange(int(self.y_plot_range[0]), int(self.y_plot_range[1]) + self.bin_size, self.bin_size)
        loc_dens_hist, _, _ = np.histogram2d(locs[:, 1], locs[:, 2], bins=(bin_x_edges, bin_y_edges))
        max_loc_dens = np.max(loc_dens_hist)
        plt.figure('Localization density')
        plt.hist2d(locs[:, 1], locs[:, 2], bins=(bin_x_edges, bin_y_edges), cmap='magma', vmin=max_loc_dens * 0.1)
        for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
            plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)
        plt.xlim(self.x_plot_range)
        plt.ylim(self.y_plot_range)
        # Annotations
        plt.gca().set_aspect('equal'), plt.xlabel('x (nm)'), plt.ylabel('y (nm)'), plt.tight_layout()
        plt.show()

    def plot_locs_withcrb(self, locs):
        """
        This function plots the localizations superimposed with the CRB map
        """
        # Create the CRB plot with the same extent as the scatter plots
        plt.figure('CRB_map with loalizations')
        # Shift CRB map by the same amount as the EBP
        plt.imshow(
            self.σ_CRB, cmap='viridis', vmin=0, vmax=20,
            extent=(
                - self.ebp.pos_mins_nm[0][0] - 0.5, self.σ_CRB.shape[1] - self.ebp.pos_mins_nm[0][0] - 0.5,
                - self.ebp.pos_mins_nm[0][1] - 0.5, self.σ_CRB.shape[0] - self.ebp.pos_mins_nm[0][1] - 0.5
            ),
            origin='lower'
        )
        plt.colorbar(label='σ_CRB Value')

        # Plot PSF minima positions with the same color mapping as before
        for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
            plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)

        # add localizations to plot
        plt.scatter(locs[:, 1], locs[:, 2], c='gray', s=20, alpha=0.05)
        
        # Ensure the axes and aspect ratio are the same as in scatter plots
        plt.gca().set_aspect('equal')
        plt.xlim(self.x_plot_range)
        plt.ylim(self.y_plot_range)
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.title('σ_CRB with Aligned Reference Frame')
        plt.tight_layout()
        plt.show()

    def plot_locs_gmm_filtered(self, locs, n_components=20, sigma_threshold=1.5):
        """
        Plots two separate figures:
        - One with time-encoded localizations.
        - One with cloud numbers (labels) overlaid on the centers of each cloud.
        - Additionally, a third figure comparing the localizations with their covariance ellipses.
        """
        # Extraer coordenadas y tiempos
        x, y = locs[:, 1], locs[:, 2]
        times = locs[:, 0] - locs[0, 0]  # Tiempo relativo
        data = np.column_stack((x, y))
    
        # Preprocesamiento con KMeans para inicializar los centros
        kmeans = KMeans(n_clusters=n_components, random_state=42)
        kmeans.fit(data)
    
        # Ajustar GMM con n_components y centros iniciales de KMeans
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42, 
                              init_params='kmeans')
        gmm.fit(data)
        labels = gmm.predict(data)
    
        # Obtener medias y desviaciones estándar
        means = gmm.means_
        covariances = gmm.covariances_
        sigmas = np.sqrt(np.array([np.diag(cov) for cov in covariances]))  # σ_x, σ_y
    
        # Calcular el promedio de tiempo para cada nube (grupo)
        avg_times = []
        for i in range(n_components):
            mask = (labels == i)
            avg_times.append(np.mean(times[mask]))
    
        # Ordenar los índices de las nubes por el promedio de tiempo
        sorted_indices = np.argsort(avg_times)
    
        # Imprimir estadísticos de cada nube, ordenadas por tiempo promedio
        print("Estadísticos de cada nube (ordenadas por tiempo promedio):")
        for idx in sorted_indices:
            mu_x, mu_y = means[idx]
            sigma_x, sigma_y = sigmas[idx]
            sigma_magnitude = np.sqrt(sigma_x**2 + sigma_y**2)
            print(f"Nube {idx+1}:")
            print(f"  Media (mu_x, mu_y): ({mu_x:.2f}, {mu_y:.2f})")
            print(f"  Desviación estándar (sigma_x, sigma_y): ({sigma_x:.2f}, {sigma_y:.2f})")
            print(f"  Módulo de sigma: {sigma_magnitude:.2f}\n")
    
        # Filtrar puntos dentro de 'sigma_threshold' sigmas de su centro
        filtered_points = []
        filtered_times = []
        for i in range(n_components):
            mu_x, mu_y = means[i]
            sigma_x, sigma_y = sigmas[i]
    
            # Extraer puntos del cluster
            mask = (labels == i)
            x_cluster, y_cluster, t_cluster = x[mask], y[mask], times[mask]
    
            # Filtrar por sigma_threshold
            valid_mask = (
                (np.abs(x_cluster - mu_x) < sigma_threshold * sigma_x) & 
                (np.abs(y_cluster - mu_y) < sigma_threshold * sigma_y)
            )
            filtered_points.append(np.column_stack((x_cluster[valid_mask], y_cluster[valid_mask])))
            filtered_times.append(t_cluster[valid_mask])
    
        # Unir puntos filtrados
        filtered_data = np.vstack(filtered_points)
        filtered_times = np.concatenate(filtered_times)
    
        # Figura 1: Gráfico con codificación de tiempo
        plt.figure('GMM-Filtered Time-Encoded Localizations')
        # Superponer mínimos del EBP si están disponibles
        if hasattr(self, 'ebp') and hasattr(self.ebp, 'pos_mins_centered_nm'):
            for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
                plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)
        plt.scatter(filtered_data[:, 0], filtered_data[:, 1], c=filtered_times, cmap='rainbow', s=20, alpha=0.05)
    
        # Barra de color
        color_bar = plt.colorbar(label="Time [s]", orientation="vertical")
        color_bar.solids.set(alpha=1)
    
        # Ajustar ejes
        plt.xlim(self.x_plot_range)
        plt.ylim(self.y_plot_range)
        plt.gca().set_aspect('equal')
        plt.xlabel('x (nm)', fontsize=14)
        plt.ylabel('y (nm)', fontsize=14)
        #plt.title('GMM-Filtered Time-Encoded Localizations', fontsize=16)
        plt.tight_layout()
        plt.show()
    
        # Figura 2: Gráfico con numeración de las nubes
        plt.figure('GMM-Filtered Clouds with Labels')
    
        # Graficar los centros de las nubes y etiquetarlas
        for idx in range(n_components):
            mu_x, mu_y = means[idx]
            plt.scatter(mu_x, mu_y, color='steelblue', label=f'Nube {idx+1}', s=30)  # Centro de la nube
            plt.text(mu_x + 5, mu_y + 5, f'{idx+1}', color='black', fontsize=12)  # Número de la nube
    
        # Superponer mínimos del EBP si están disponibles
        if hasattr(self, 'ebp') and hasattr(self.ebp, 'pos_mins_centered_nm'):
            for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
                plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)
    
        # Ajustar ejes
        plt.xlim(self.x_plot_range)
        plt.ylim(self.y_plot_range)
        plt.gca().set_aspect('equal')
        plt.xlabel('x (nm)', fontsize=14)
        plt.ylabel('y (nm)', fontsize=14)
        plt.title('GMM-Filtered Clouds with Labels', fontsize=16)
        plt.tight_layout()
        plt.show()
    
        # Figura 3: Comparación de las nubes y las elipses
        plt.figure('Comparison of Clouds with Ellipses and Localizations')
        for idx in range(n_components):
            mu_x, mu_y = means[idx]
            sigma_x, sigma_y = sigmas[idx]
            
            # Calcular el módulo de sigma
            sigma_magnitude = np.sqrt(sigma_x**2 + sigma_y**2)
    
            # Extraer las localizaciones de la nube
            mask = (labels == idx)
            x_cluster, y_cluster = x[mask], y[mask]
    
            # Graficar las localizaciones de la nube
            plt.scatter(x_cluster, y_cluster, color='powderblue', s=10, alpha=0.3)
    
            # Calcular la matriz de covarianza y la orientación
            covariance = covariances[idx]
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            axis_lengths = np.sqrt(eigenvalues) * 1.52  # Ajuste del factor de escala
            orientation = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])  # Ángulo de rotación
    
            # Graficar la elipse representativa de la nube
            ellipse = plt.matplotlib.patches.Ellipse(
                (mu_x, mu_y),  # Centro de la elipse
                width=axis_lengths[0],  # Eje mayor
                height=axis_lengths[1],  # Eje menor
                angle=np.degrees(orientation),  # Ángulo de rotación
                edgecolor='steelblue',  # Color del borde
                facecolor='none',  # Sin relleno
                lw=2  # Grosor del borde
            )
            plt.gca().add_patch(ellipse)
            plt.scatter(mu_x, mu_y, color='steelblue', s=20)  # Centro de la nube
    
        # Añadir los mínimos del EBP si están disponibles
        # if hasattr(self, 'ebp') and hasattr(self.ebp, 'pos_mins_centered_nm'):
        #     for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
        #         plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)
        # Añadir etiquetas y personalizar el gráfico
        plt.xlabel('x (nm)', fontsize=14)
        plt.ylabel('y (nm)', fontsize=14)
        #plt.title('Comparison of Clouds with Ellipses and Localizations', fontsize=16)
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()

    # Figura 4: Elipses representando la covarianza de las nubes
        plt.figure('Covariance Ellipses')
        for idx in range(n_components):
            mu_x, mu_y = means[idx]
            sigma_x, sigma_y = sigmas[idx]
            
            # Calcular el módulo de sigma
            sigma_magnitude = np.sqrt(sigma_x**2 + sigma_y**2)
    
            # Extraer las localizaciones de la nube
            mask = (labels == idx)
            x_cluster, y_cluster = x[mask], y[mask]
    
            # Graficar las localizaciones de la nube
            #plt.scatter(x_cluster, y_cluster, color='powderblue', s=10, alpha=0.3)
    
            # Calcular la matriz de covarianza y la orientación
            covariance = covariances[idx]
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            axis_lengths = np.sqrt(eigenvalues) * 1.52  # Ajuste del factor de escala para que incluya el 39% de la región
            orientation = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])  # Ángulo de rotación
    
            # Graficar la elipse representativa de la nube
            ellipse = plt.matplotlib.patches.Ellipse(
                (mu_x, mu_y),  # Centro de la elipse
                width=axis_lengths[0],  # Eje mayor
                height=axis_lengths[1],  # Eje menor
                angle=np.degrees(orientation),  # Ángulo de rotación
                edgecolor='steelblue',  # Color del borde
                facecolor='none',  # Sin relleno
                lw=2  # Grosor del borde
            )
            plt.gca().add_patch(ellipse)
            plt.scatter(mu_x, mu_y, color='steelblue', s=20)  # Centro de la nube
    
            # Imprimir estadísticos de cada nube
            print(f"Nube {idx+1}:")
            print(f"  Media (mu_x, mu_y): ({mu_x:.2f}, {mu_y:.2f})")
            print(f"  Desviación estándar (sigma_x, sigma_y): ({sigma_x:.2f}, {sigma_y:.2f})")
            print(f"  Módulo de sigma: {sigma_magnitude:.2f}\n")
            # Añadir los mínimos del EBP si están disponibles
        if hasattr(self, 'ebp') and hasattr(self.ebp, 'pos_mins_centered_nm'):
            for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
                plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)
        # Añadir etiquetas y personalizar el gráfico
        plt.xlabel('x (nm)', fontsize=14)
        plt.ylabel('y (nm)', fontsize=14)
        plt.title('Covariance Ellipses', fontsize=16)
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()
    
