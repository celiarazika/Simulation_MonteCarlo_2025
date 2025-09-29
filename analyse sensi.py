import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_parameter_influence(param_name, param_values, base_params, n_simul=100):
    results = []
    half_widths = []
    
    for value in tqdm(param_values, desc=f"Analyse de {param_name}"):
        params = base_params.copy()
        
        if param_name == "x0" or param_name == "eta":
            params[param_name] = value
        elif param_name == "pP":
            params["pacc"] = (params["pacc"][0], params["pacc"][1], value)
        elif param_name == "pNP":
            pSN, pNS, _, pPN = params["ptrans"]
            params["ptrans"] = (pSN, pNS, value, pPN)
        elif param_name == "s":
            params["s"] = value
        
        result = msa(n_simul, params)
        
        results.append(result["s"])
        half_widths.append(result["demi_largeur"])
    
    return param_values, results, half_widths

base_params = {
    "s": 50,
    "N": 100,
    "pacc": (0.01, 0.05, 0.1),
    "ptrans": (0.3, 0.4, 0.2, 0.5),
    "alpha": 1,
    "x0": 2,
    "eta": 2
}

x0_values = np.linspace(1, 5, 10)
eta_values = np.linspace(1.6, 3.0, 10)
pP_values = np.linspace(0.05, 0.3, 10)
pNP_values = np.linspace(0.1, 0.5, 10)
s_values = np.linspace(20, 100, 10)

x0_vals, x0_results, x0_half_widths = analyze_parameter_influence("x0", x0_values, base_params)
eta_vals, eta_results, eta_half_widths = analyze_parameter_influence("eta", eta_values, base_params)
pP_vals, pP_results, pP_half_widths = analyze_parameter_influence("pP", pP_values, base_params)
pNP_vals, pNP_results, pNP_half_widths = analyze_parameter_influence("pNP", pNP_values, base_params)
s_vals, s_results, s_half_widths = analyze_parameter_influence("s", s_values, base_params)

plt.figure(figsize=(20, 15))
plt.suptitle("Influence des parametres sur m(s)", fontsize=16)

plt.subplot(2, 3, 1)
plt.errorbar(x0_vals, x0_results, yerr=x0_half_widths, marker='o', linestyle='-', capsize=5)
plt.xlabel("Parametre x0")
plt.ylabel("m(s)")
plt.title("Influence de x0")
plt.grid(True)

plt.subplot(2, 3, 2)
plt.errorbar(eta_vals, eta_results, yerr=eta_half_widths, marker='o', linestyle='-', capsize=5)
plt.xlabel("Parametre eta")
plt.ylabel("m(s)")
plt.title("Influence de eta")
plt.grid(True)

plt.subplot(2, 3, 3)
plt.errorbar(pP_vals, pP_results, yerr=pP_half_widths, marker='o', linestyle='-', capsize=5)
plt.xlabel("Probabilite d'accident par temps pluvieux (pP)")
plt.ylabel("m(s)")
plt.title("Influence de pP")
plt.grid(True)

plt.subplot(2, 3, 4)
plt.errorbar(pNP_vals, pNP_results, yerr=pNP_half_widths, marker='o', linestyle='-', capsize=5)
plt.xlabel("Probabilite de transition nuages -> pluie (pNP)")
plt.ylabel("m(s)")
plt.title("Influence de pNP")
plt.grid(True)

plt.subplot(2, 3, 5)
plt.errorbar(s_vals, s_results, yerr=s_half_widths, marker='o', linestyle='-', capsize=5)
plt.xlabel("Seuil s")
plt.ylabel("m(s)")
plt.title("Influence de s")
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('influence_parametres.png', dpi=300)
plt.show()
