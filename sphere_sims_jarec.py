from utils import *
from scipy.stats import ortho_group
import sys
import argparse
parser = argparse.ArgumentParser(description='Description of your script')

# Add command line arguments
parser.add_argument('--n_2', type=int, help='Value of n_2')
parser.add_argument('--type', type=str, help='Type of simulation (center, radius, V)')

# Parse the command line arguments
args = parser.parse_args()

# Access the values of n_2 and type
n_2 = args.n_2
type_value = args.type
print(type_value + ' sample size: ' + str(n_2))
B=500
D = 10
d = 2
n = 1000

c_1 = np.zeros(D)
r_1 = 100

V = ortho_group.rvs(dim=D)
V_1 = V[:,0:d+1]
X_1 = pd.DataFrame(generate_data_on_sphere(V_1, c_1, r_1, D, d, n))

def evaluate_sims(X_1, V_1, c_1, r_1, n_2, filename, type_value):

    num_runs =100

    ##CENTERS
    if (type_value == "center"):
        centers = [round(x, 2) for x in list(np.arange(-5, -3, 0.75)) + list(np.arange(-3, 3, 0.25)) + list(np.arange(3, 5.5, 0.75))]
        boots_c = []
        boots_ss_c = []
        sss_c = []


        for center_shift in tqdm(centers, desc='Evaluating Centers', leave=True):
            avg_boots = 0
            avg_boots_ss = 0
            avg_sss = 0
            
            for _ in tqdm(range(num_runs), desc='Center run', leave=False): 
                r = 0
                V_2, c_2, r_2 = shift_params(V_1, c_1, r_1, 0, center_shift, 0)
                X_2 = pd.DataFrame(generate_data_on_sphere(V_2, c_2, r_2, D, d, n_2, seed = r))
                
                d_null_boot, d_test_boot, ss_boot = HypTestBootAll(X_1, X_2, B=B, d=2, multi=True)
                ss_result = ss_test(X_1, X_2, d=2)

                avg_boots += np.mean(d_test_boot / d_null_boot)
                avg_boots_ss += np.mean(ss_boot)
                avg_sss += ss_result
                r += 1

            # Calculate the average for each metric
            avg_boots /= num_runs
            avg_boots_ss /= num_runs
            avg_sss /= num_runs

            boots_c.append(avg_boots)
            boots_ss_c.append(avg_boots_ss)
            sss_c.append(avg_sss)

        results_centers = pd.DataFrame({'centers': centers, 'boots_c': boots_c, 'boots_ss_c': boots_ss_c, 'sss_c': sss_c})
        results_centers.to_csv("results/"+filename+"center_results.csv")

    ##V SHIFT
    if (type_value == "V"):
        rotations = list(range(0, 361, 20))
        boots_v = []
        boots_ss_v = []
        sss_v = []

        for v_shift in tqdm(rotations, desc='Evaluating V', leave=True):
            avg_boots = 0
            avg_boots_ss = 0
            avg_sss = 0
            
            
            for _ in tqdm(range(num_runs), desc='V run', leave=False):
                r=0
                
                V_2, c_2, r_2 = shift_params(V_1, c_1, r_1, v_shift, 0, 0)
                X_2 = pd.DataFrame(generate_data_on_sphere(V_2, c_2, r_2, D, d, n_2, seed = r))

                d_null_boot, d_test_boot, ss_boot = HypTestBootAll(X_1, X_2, B=B, d = 2, multi=True)
                
                ss_result = ss_test(X_1, X_2, d=2)

                avg_boots += np.mean(d_test_boot / d_null_boot)
                avg_boots_ss += np.mean(ss_boot)
                avg_sss += ss_result
                r += 1

            # Calculate the average for each metric
            avg_boots /= num_runs
            avg_boots_ss /= num_runs
            avg_sss /= num_runs

            boots_v.append(avg_boots)
            boots_ss_v.append(avg_boots_ss)
            sss_v.append(avg_sss)     

        results_v = pd.DataFrame({'rotations': rotations, 'boots_v': boots_v, 'boots_ss_v': boots_ss_v,'sss_v': sss_v})
        results_v.to_csv("results/"+filename+"V_results.csv")

    ##RADII
    if (type_value == "radius"):
        radii = [round(x, 2) for x in list(np.arange(0.1, 0.8, 0.1)) + list(np.arange(0.8, 1.2, 0.05)) + list(np.arange(1.2, 2.1, 0.1))]
        boots_r = []
        boots_ss_r = []
        sss_r = []


        for radius_shift in tqdm(radii, desc='Evaluating Radius', leave=True):
            avg_boots = 0
            avg_boots_ss = 0
            avg_sss = 0
            for _ in tqdm(range(num_runs), desc='Radius run', leave=False):
                r = 0
                r_scaled = (radius_shift * r_1) - r_1
                V_2, c_2, r_2 = shift_params(V_1, c_1, r_1, 0, 0, r_scaled)
                X_2 = pd.DataFrame(generate_data_on_sphere(V_2, c_2, r_2, D, d, n_2, seed = r))

                d_null_boot, d_test_boot, ss_boot = HypTestBootAll(X_1, X_2, B=B, d=2, multi=True)
                ss_result = ss_test(X_1, X_2, d=2)

                avg_boots += np.mean(d_test_boot / d_null_boot)
                avg_boots_ss += np.mean(ss_boot)
                avg_sss += ss_result
                r += 1

            # Calculate the average for each metric
            avg_boots /= num_runs
            avg_boots_ss /= num_runs
            avg_sss /= num_runs

            boots_r.append(avg_boots)
            boots_ss_r.append(avg_boots_ss)
            sss_r.append(avg_sss)

        results_radius = pd.DataFrame({'radii': radii, 'boots_r': boots_r, 'boots_ss_r' : boots_ss_r, 'sss_r': sss_r})
        results_radius.to_csv("results/"+filename+"radius_results.csv")


evaluate_sims(X_1, V_1, c_1, r_1, n_2=n_2, type_value = type_value, filename="sphere_n_" + str(n_2) + "_")

