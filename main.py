from scipy import integrate, signal, special
from math import sqrt, log10, exp, pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rc('font', family='serif')
mpl.rcParams.update({'font.size': 12})
mpl.rcParams.update({'legend.labelspacing':0.25, 'legend.fontsize': 12})
mpl.rcParams.update({'errorbar.capsize': 4})

#input parameter - magnetic field of magnetar
B =  4*1e13

# From the article Gullon et al.(2015)
beta_T = 0.3
#tau_res = B/1e14

#Use this formula for magnetar's model. For check - tau_0 = number
#tau_0 = 2*beta_T*tau_res
tau_0 = 1

def flux_n_tr(k):
	"""
	Calculate the transmitted flux of resonant cyclotron scattering, see M.Lyutikov & F.P.Gavriil(2006)
	Input parameter: relation omega/omega_0
	"""
	omega_0 = 8
	omega = k*omega_0

	# another way to find omega_0 independly
	#omega = 0.1
	#omega_0 = omega/k

	dif = omega - omega_0
	if k == 1:	
		n_transm = (exp(-tau_0/2)*(10*signal.unit_impulse(2)[0] + tau_0/(8*beta_T*omega_0)*sqrt((omega_0*(1 + 4*beta_T) - omega)/(dif + 0.1))* 
			special.i1((tau_0/(4*beta_T*omega_0)*sqrt(dif*(omega_0*(1 + 4*beta_T) - omega))))))
	elif  1 < k <= 1.4: 
		n_transm = (exp(-tau_0/2)*(signal.unit_impulse(2)[1] + tau_0/(8*beta_T*omega_0)*sqrt((omega_0*(1 + 4*beta_T) - omega)/dif)*
			special.i1((tau_0/(4*beta_T*omega_0)*sqrt(dif*(omega_0*(1 + 4*beta_T) - omega))))))
	else:
		n_transm = 0
	return n_transm

def flux_n_ref(k):
	"""
	Calculate the reflected flux of resonant cyclotron scattering, see M.Lyutikov & F.P.Gavriil(2006)
	Input parameter: relation omega/omega_0
	"""
	omega_0 = 8
	omega = k*omega_0

	# another way to find omega_0 independly
	#omega = 0.1
	#omega_0 = omega/k

	if 0.8 <= k <= 1.2:
		n_reflect = (tau_0/(8*beta_T*omega_0)*exp(-tau_0/2)*
			special.i0(tau_0/(2*beta_T*omega_0)*sqrt((omega_0*(1 + 2*beta_T) - omega)*(omega - omega_0*(1 - 2*beta_T)))))
	else:
		n_reflect = 0
	return n_reflect

def flux_bb(E):
	# Blackbody
	kT = 1 #[keV]
	h = 7e-19 #this is h/2pi in [keV*s]
	c = 3e10 # [сm/s]
	B = E**3/(4.0*pi**3*c**2*h**3)*(1/(exp(E/kT)-1))
	return B


if __name__ == "__main__":
	#remember, that you should change omega_0 = [from 0.1 to 10] dependig from tau_0
	#Use the best fitting with an article Gullon et al.(2015) Fig.2 (spectrum)
	
	h = 7e-19 #Planck constant (h/2pi) in [keV*s]
#-----------------------------------------------------
	#n_+, matrix
	n_tr, X = [], []
	c = 0 #counter of rows
	for j in np.arange(0.1,10,0.02): #Energy keV
		n_tr.append([]) #add row
		for i in np.arange(0.1,10,0.02): #fill row
			k = j/i # relation
			n_tr[c].append(flux_n_tr(k))
		c += 1		#next row
		X.append(j)
	n_plus = np.array(n_tr)
#----------------------------------------------------	
	#n_-, matrix
	n_ref = []
	c = 0 #counter of rows
	for j in np.arange(0.1,10,0.02): #Energy keV
		n_ref.append([]) #add row
		for i in np.arange(0.1,10,0.02): #fill row
			k = j/i # relation
			n_ref[c].append(flux_n_ref(k))
		c += 1		#next row
	n_minus = np.array(n_ref)

#----------------------------------------------------
	#Blackbody, vector
	Bb = []
	for j in np.arange(0.1,10,0.02): #Energy keV
		Bb.append(flux_bb(j))
	blackbody = np.array(Bb)
	blackbody = blackbody.reshape(-1,1) #row vector to column vector
#----------------------------------------------------
	#first integral
	integ1 = np.matmul(n_plus, blackbody)
	integ1 = 0.02/h*integ1
	#print(integ1[1][1])
#----------------------------------------------------
	#second integral - 23.3% from first integral
	#interior
	integ2_1 = np.matmul(n_minus, blackbody)
	integ2_1 = 0.02/h*integ2_1
	#external
	integ2_2 = np.matmul(n_plus, integ2_1)
	integ2_2 = 0.02*integ2_2

#----------------------------------------------------
	#third integral
	result = []
	integ3_1 = integ2_2
	integ3_2 = 0.02*np.matmul(n_minus, integ3_1)
	integ3_3 = 0.02*np.matmul(n_plus, integ3_2)
	result = integ1 + integ2_2 + integ3_3
#-----------------------------------------------------

plt.plot(X, result/np.sum(result), linewidth=1, label=r'$Model,\qquad \tau_0 = 1 $')
plt.plot(X, Bb/np.sum(Bb), label=r'$Blackbody$')
plt.legend()
plt.show()
#plt.savefig('tau_1.pdf')