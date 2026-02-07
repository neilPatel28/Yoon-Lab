import numpy as np
import matplotlib.pyplot as plt
import tauc_lorentz_model as tlm
import intialGuessStartingPoint as gs
import fittingTool as ft


#the point of this program is to check if the model is correct
# we do that by
# first creating data that matches our experimental data
# then we fit the data
# check if the fitted params match the input params

#raw data
#from the experiment we can see that at cold temperatures our exciton width becomes 0.007eV
ws2_exp_params = np.array([1.40297001e+02, 1.15183473e+01,
                           2.95616242e+00, 2.92483712e-01,4.27646826e-01, 1.99807325e+00,
                            # change to match the experiment
                           1.97, # E0
                           4.71331035e-01,
                           0.007, #G0
                           1.80029956e+00,
                           2.37185291e+00, 1.18930408e-01, 2.28297482e-01, 1.87449362e+00,
                           2.69456857e+00, 1.17254844e-01, 2.23229104e-01, 1.73045554e+00])



wse2_inx_params = np.array([3.55056955e+01, 4.78836239e+00,
                            2.90640824e+00, 5.81521714e-01, 5.29399239e-01, 3.99176390e-01 ,
                            2.42574793e+00, 1.44599416e-01, 3.55051370e-01 ,1.19803598e-12,
                            1.66965808e+00 ,2.16462602e-01, 6.43666142e-02, 1.03755299e+00,
                            2.08861547e+00, 5.74162675e-02, 2.49299310e-01 ,5.82112101e-11])


wse2_exp_params = np.array([3.55056955e+01, 4.78836239e+00,
                            2.90640824e+00, 5.81521714e-01, 5.29399239e-01, 3.99176390e-01
                               ,2.42574793e+00, 1.44599416e-01,3.55051370e-01 ,1.19803598e-12,
                            1.7+00 ,2.16462602e-01,0.007,
                            1.03755299e+00, 2.08861547e+00, 5.74162675e-02, 2.49299310e-01 ,5.82112101e-11])


ws2_inx_params = np.array([1.40297001e+02, 1.15183473e+01,
                           2.95616242e+00, 2.92483712e-01,4.27646826e-01, 1.99807325e+00,
                           1.97311076e+00, 4.71331035e-01, 5.82731386e-02, 1.80029956e+00,
                           2.37185291e+00, 1.18930408e-01, 2.28297482e-01, 1.87449362e+00,
                           2.69456857e+00, 1.17254844e-01, 2.23229104e-01, 1.73045554e+00])
#good

#change between ws2 and wse2
#ws2_exp_params = wse2_exp_params
#change data too
#ws2_inx_params = wse2_inx_params
data = np.loadtxt(
        r"ws2index.txt"
    )

#grab eV spectrum


wavelengths = data[:, 0]
n = data[:, 1]
k = data[:, 2]
wavelengths = wavelengths * 1e-6
energy = 0.00000123982884337/wavelengths

N = n + 1j * k
er = N ** 2


#find the er for the params
tl_model = tlm.TaucLorentz()
er_model_exp = tl_model.Lorentz_oscillator_model(energy, *ws2_exp_params)
er_model_index = tl_model.Lorentz_oscillator_model(energy, *ws2_inx_params)


# Plot
plt.figure(figsize=(12, 5))

# Real part
plt.subplot(1, 2, 1)
plt.plot(energy, er.real, 'b.', label='raw')
plt.plot(energy, er_model_exp.real, 'r-', label=' new Re(ε) model')
#plt.plot(energy, er_model_index.real, 'g-', label='Re(ε) model')
plt.xlabel("Energy (eV)")
plt.ylabel("Re(ε)")
plt.title("Real Permittivity")
plt.legend()

# Imaginary part
plt.subplot(1, 2, 2)
plt.plot(energy, er.imag, 'b.', label='raw')
plt.plot(energy, er_model_exp.imag, 'r-', label='new Im(ε) model')
#plt.plot(energy, er_model_index.imag, 'g-', label='Re(ε) model')
plt.xlabel("Energy (eV)")
plt.ylabel("Im(ε)")
plt.title("Imaginary Permittivity")
plt.legend()

plt.tight_layout()
plt.show()

print("new data vs raw file(graph 1)")
print(" ")
print(" ")
print(" ")
print(" ")











print("old fit vs raw (graph 2) (the old fit)")
popt = ws2_inx_params
# Print parameters
print("Scale and Add:", popt[:2])
osc_params = popt[2:]
n_osc = len(osc_params) // 4
for i in range(n_osc):
    start = i * 4
    end = start + 4
    print(f"Oscillator {i + 1}:", osc_params[start:end])

# Plot
plt.figure(figsize=(12, 5))

# Real part
plt.subplot(1, 2, 1)
plt.plot(energy, er.real, 'b.', label='raw')
plt.plot(energy, er_model_index.real, 'g-', label='original fit: Re(ε) model')
plt.xlabel("Energy (eV)")
plt.ylabel("Re(ε)")
plt.title("Real Permittivity")
plt.legend()

# Imaginary part
plt.subplot(1, 2, 2)
plt.plot(energy, er.imag, 'b.', label='raw')
plt.plot(energy, er_model_index.imag, 'g-', label='original fit: Imag(ε) model')
plt.xlabel("Energy (eV)")
plt.ylabel("Im(ε)")
plt.title("Imaginary Permittivity")
plt.legend()

plt.tight_layout()
plt.show()











print(" ")
print(" ")
print(" ")
print(" ")







print("OUTPUT Params vs self (graph 3) fitted against self")
guess = gs.InitialGuessCreator(None,energy,er_model_exp)
directguess = guess.create_guess()
fitter = ft.Fitter(None,energy,er_model_exp)

# Run fit
popt = fitter.fit(ws2_exp_params)
# Print parameters
print("Fitted parameters[Ei,Ai,Ci(width),Eg]:")
print("Scale and Add:", popt[:2])
osc_params = popt[2:]
n_osc = len(osc_params) // 4
for i in range(n_osc):
    start = i * 4
    end = start + 4
    print(f"Oscillator {i + 1}:", osc_params[start:end])
print(popt)
# Plot results
fitter.plot_fit()
