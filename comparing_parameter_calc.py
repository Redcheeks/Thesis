import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate a sample distribution of rheobase currents (I_distr) in nA
I_distr = np.linspace(1, 100, 300) * 1e-9  # Convert to A

# Approach 1: I → S → R
S_from_I = 3.96e-4 * (I_distr**0.396)  # Soma size from Table 4
R_from_S = (1.68e-10) / (S_from_I**2.43)  # Resistance from Table 4

# Approach 2: I → R directly
R_direct = 3.1e-2 * (I_distr**-0.96)  # Resistance from Table 3

# Compare the two results
df_comparison = pd.DataFrame(
    {
        "I_distr (A)": I_distr,
        "S_from_I (m^2)": S_from_I,
        "R_from_S (Ω)": R_from_S,
        "R_direct (Ω)": R_direct,
        "Difference (Ω)": R_from_S - R_direct,
    }
)

# Save the comparison to a CSV file
# csv_filename = "I_to_R_comparison.csv"
# df_comparison.to_csv(csv_filename, index=False)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(
    I_distr * 1e9, R_from_S, label="R from S (I → S → R)", linestyle="--", color="blue"
)
plt.plot(
    I_distr * 1e9,
    R_direct,
    label="R from I directly (I → R)",
    linestyle="-",
    color="red",
)
plt.xlabel("Rheobase Current (nA)")
plt.ylabel("Resistance (Ω)")
plt.title("Comparison of Resistance Calculations")
plt.legend()
plt.grid(True)
plt.show()

# Display the first few rows of the comparison table
df_comparison.head()
