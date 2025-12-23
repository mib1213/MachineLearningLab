import numpy as np
import shap

def f(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    return 5.10 + 1.20 * x1 - 0.80 * x2 + 2.00 * x3 + 0.50 * np.sin(x1) + 0.30 * x1 * x3 - 0.20 * (x2 ** 2)

D = np.array([
    [1.2, -0.4, 0],
    [2.8,  1.3, 1],
    [0.5,  0.7, 0],
    [3.1, -1.1, 1],
    [1.9,  2.0, 1],  # x (zu erklären)
    [2.2,  0.2, 0],
    [0.9,  1.6, 0],
], dtype=float)

# Background B = {1,3,4,6}:
B_idx = [0, 2, 3, 5]
B = D[B_idx]

# zu erklärender Punkt x = k=5:
x = D[4:5]  # shape (1, 3)


masker = shap.maskers.Independent(B)
explainer = shap.Explainer(f, masker, algorithm="exact")

exp = explainer(x)

phi = exp.values[0]         # Shapley-Werte pro Feature für x
base = exp.base_values[0]   # entspricht f(∅) = Mittelwert über Background (bei Independent)
fx = f(x)[0]

print("SHAP ExactExplainer")
print(f"Erklärter Punkt x        = ({x[0,0]:.1f}, {x[0,1]:.1f}, {x[0,2]:.1f})")
print(f"Modellwert f(x)          = {fx:.3f}")
print(f"Baseline f(∅)            = {base:.3f}")
print()
print("Shapley-Werte:")
print(f"  φ_X1 = {phi[0]: .3f}")
print(f"  φ_X2 = {phi[1]: .3f}")
print(f"  φ_X3 = {phi[2]: .3f}")
print()
print(f"Check: f(∅) + φ_X1 + φ_X2 + φ_X3 = {(base + phi.sum()):.3f}")