from math import sqrt
from math import exp
from math import tan
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.special import yn
from scipy.special import jv
import scipy.special as sp
from scipy.optimize import fsolve
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.style.use("journalPlot")


# ------------------------- Constants Definition -----------------------------------
hbar = 6.582e-16  # Planck's constant (JeV·s)
me = 9.10938356e-31    # Electron mass (kg)


# ------------------------- Geometric Variable Definition --------------------------
#Hardcoded for now, should be pulled from GUI
Lz = 10            # Length of the cylinder in z-direction (nm)
R = 5              # Radius of the quantum dot (nm)
V0 = 1.6           # Finite potential outside the well (eV)

#---------------------------- Material Class definitions ------------------------------
'''
Materials are defined as a class, with specific instances pre-defined to populate the GUI with common semiconductors
'''
class Material:
    def __init__(self, name, effectiveMassE, effectiveMassH, bandGap):
        # Initialize the material's properties: electron and hole effective mass, and bandgap
        self.name = name
        self.effectiveMassE = effectiveMassE 
        self.effectiveMassH = effectiveMassH
        self.bandGap = bandGap 

    def __str__(self):
        # Gives the ability to print a material's properties
        return (f"{self.name} Material Properties:\n"
                f"  Effective Eletron Mass: {self.effectiveMassE} kg\n"
                f"  Effective Hole Mass: {self.effectiveMassH} kg\n"
                f"  Band Gap: {self.bandGap} eV\n")
        
# Pre-define library of standard materials 
GaAs = Material(name = "Gallium Arsenide",
                effectiveMassE=0.067 * 9.10938356e-31,
                effectiveMassH=0.067 * 9.10938356e-31,
                bandGap=1.42)
# Add library to a list for ease of iteration 
materialList = [
    GaAs
]


# ------------------------- Potential Well Definitions -----------------------------
def calc_bandgap(coreBg = 1.42, shellBg = 2.24, bgRatio = 0.6):  
    '''
    Takes in the typical band gap of the core and shell materials and returns the potential barrier in the valence and conduction bands, according to some ratio
    Eg. a band gap may be weighted 60% to the conduction band and 40% to the valence band for a GaP/GaAs QD (this is the default case if no value is provided)
    Strain considerations need to be implemented.
    '''
    conductionBg = bgRatio*(shellBg-coreBg)
    valence_bg = (1-bgRatio)*(shellBg-coreBg)
    return (conductionBg, valence_bg)

# ------------------------- Infinite Solution -----------------------------------
def solve_radial_infinite(R, k, m):
    zeros = sp.jn_zeros(int(m), 10)  #find the first 10 zeros
    Er = (hbar**2 * zeros[k]**2) / (2 * me * R**2) 

    return Er

# Axial equation
def solve_axial_infinite(n, L):
    Ez = (n**2 * np.pi**2 * hbar**2) / (2 * me * L**2)
    return Ez

# Find a series of eigenenergies
def solve_quantum_dot_infinite():
    '''
    Iterates through quantum numbers n, m, and k to find allowed energy levels that are less than the V0. 
    '''
    E = []
    for n in range(1,10):
        for m in range(1,10):
            for k in range(1,10):
                Etot = solve_axial_infinite(n, Lz) + solve_radial_infinite(R, k,m) # Energy is the linear combination of axial and radial energies
                if Etot > calc_bandgap()[1]: 
                    print("Max E: " + str(Etot)) # Omit solutions that would fall outside the well
                    break
                print(E)
                E.append(Etot)
    return E

def plot_infinite():
    E = solve_quantum_dot_infinite()
    V = calc_bandgap()[1]
    
    # Add lines to represent the potential well
    plt.vlines([0.5, -0.5], 0, V, 'k')
    plt.hlines(V, -1, -0.5, 'k')
    plt.hlines(0, -0.5, 0.5, 'k')
    plt.hlines(V, 0.5, 1, 'k')

    # Plot allowed energies
    plt.hlines(E, -0.5, 0.5,)
    plt.title("Eigenenergies of an Infinite Cylindrical QD")
    plt.xlabel("")
    plt.ylabel("Energy (eV)")
    plt.show()

#------------------------------- Finite Solution ---------------------------------

def axial_equation(E, V, symmetrical): 
    '''
    Defines the equation for energy for symmetric and asymmetric solutions to a finite potential well
    Mass needs to be changed to effective mass in the core and shell
    '''
    k = (np.sqrt(2*me*E)/hbar) # Should use effective mass in the core
    a = np.sqrt(2*me*(V-E))/hbar # Should use effective mass in the barrier
    if symmetrical:
        return  k* np.tan(k*Lz/2) - a
    else:
        return k* (1/np.tan(k*Lz/2)) - a # cot is not defined in python, use cot = 1/tan


def solve_axial_finite(V, symmetrical, tolerance=1e-6, stepSize=0.01):
    '''
    Finds all solutions to the finite potential well that fall within [0,V0]
    Tolerance accounts for FP errors and step size determines search resolution
    '''
    solutions = []
    
    for initialGuess in np.arange(0, V, stepSize): # Step through initial guesses
        solution = fsolve(axial_equation, initialGuess, args=(V, symmetrical,)) # Solutions may be symmetric or asymmetric

        # Check if either solution is unique
        if solution[0] > 0 and solution[0] < V and not any(np.abs(solution - np.array(solutions)) < tolerance):
            solutions.append(solution[0])
    
    return np.unique(np.round(solutions, decimals=6))  # Round to avoid floating-point duplicates

def radial_equation(E, V, m):
    '''
    Defines the energy equation for a circular potential well
    Mass needs to be changed to effective mass in the core and shell
    '''
    k = (np.sqrt(2*me*E)/hbar) # Should use effective mass in the core
    a = np.sqrt(2*me*(V-E))/hbar # Should use effective mass in the barrier
    return a*sp.kvp(m, a*R)*sp.jn(m, k*R)-k*sp.jvp(m, k*R)*sp.kn(m, a*R)

def solve_radial_finite(V, tolerance=1e-6, step_size=0.1):
    '''
    Finds all solutions to the finite circular well that fall within [0,V0]
    Tolerance accounts for FP errors and step size determines search resolution
    '''
    solutions = []
    
    for m in range(1, 10):
        for initalGuess in np.arange(0, V, step_size):
            solution = fsolve(radial_equation, initalGuess, args=(V, m))

            # Check if  solution is unique
            if solution[0] > 0 and solution[0] < V and not any(np.abs(solution - np.array(solutions)) < tolerance):
                solutions.append(solution[0])

    return solutions

def plot_finite_well():
    '''
    Adds the energies from the radial and axial solutions and plots them in a potential well
    '''
    V= calc_bandgap()[1]
    # Find axial solutions
    E1 = solve_axial_finite(V, True)
    E2 = solve_axial_finite(V, False)

    # Find radial solutions
    E3 = solve_radial_finite(V)

    E = [e1 + e3 for e1 in E1 for e3 in E3 if e1 + e3 < V] # Combine symmetric axial solutions with radial solutions
    E += [e2 + e3 for e2 in E2 for e3 in E3 if e2 + e3 < V] # Combine asymmetirc axial solutions with radial solutions

    fig, ax = plt.subplots(figsize=(6, 4))  # Set figure size for better fit

    # Create potential well visualization
    ax.vlines([0.5, -0.5], 0, V, 'k')
    ax.hlines(V, -1, -0.5, 'k')
    ax.hlines(0, -0.5, 0.5, 'k')
    ax.hlines(V, 0.5, 1, 'k')

    # Plot energies
    ax.hlines(E, -0.5, 0.5)
    ax.set_title("Eigenenergies of a Finite Potential Well")
    ax.set_xlabel("")
    ax.xaxis.set_visible(False) # x-axis could show units of L or R?
    ax.set_ylabel("Energy (eV)")
    ax.legend()

    # Ensure the plot clears before replotting
    for widget in framePlot.winfo_children():
        widget.destroy()

    # Embed the plot into the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=framePlot)
    canvas.draw()
    canvas.get_tk_widget().pack()



#----------------------------------- GUI ------------------------------------------------
# Handle GUI closing
def on_close():
    root.quit()
    root.destroy()

# Set up Tkinter window
root = tk.Tk()
root.title("Cylindircal QD Model")
root.protocol("WM_DELETE_WINDOW", on_close)

# Create a frame for inputs and labels
inputFrame = tk.Frame(root)
inputFrame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

# Core material drop down
coreMaterialLabel = tk.Label(inputFrame, text="Select Core:")
coreMaterialLabel.grid(row=0, column=0, sticky="w")
coreComboBox = ttk.Combobox(inputFrame, values=["GaAs", "Si", "GaP"])  # Should populate from class
coreComboBox.grid(row=0, column=1)

# Shell material drop down
shellMaterialLabel = tk.Label(inputFrame, text="Select Shell:")
shellMaterialLabel.grid(row=1, column=0, sticky="w")
shellComboBox = ttk.Combobox(inputFrame, values=["GaAs", "Si", "GaP"])  # Should populate from class
shellComboBox.grid(row=1, column=1)

# Bandgap offset manual entry
bgLabel = tk.Label(inputFrame, text="Bandgap Offset (dec):")
bgLabel.grid(row=2, column=0, sticky="w")
bgInput = tk.Entry(inputFrame)
bgInput.grid(row=2, column=1)

# Frame to display text output (could put energy values here?)
resultFrame = tk.Frame(root)
resultFrame.grid(row=0, column=1, padx=10, pady=10)

# Label for displaying results (side by side with the inputs)
result_label = tk.Label(resultFrame, text="Output data here", justify="center")
result_label.grid(row=0, column=0)

# Plot button
plot_button = tk.Button(root, text="Plot", command=plot_finite_well)
plot_button.grid(row=1, column=0, columnspan=2, pady=10)

# Frame for the plot (embedding the plot inside this frame)
framePlot = tk.Frame(root)
framePlot.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Run the application
root.mainloop()