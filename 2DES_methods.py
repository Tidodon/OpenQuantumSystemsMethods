#from msilib.schema import SelfReg
#from turtle import color
from turtle import color
from Lind2DNNmethods import Lind2DNN
import multiprocessing as mp
import numpy as np
import scipy.linalg as LA
import copy
from decimal import Decimal
import multiprocessing as mp
from itertools import product
import time
import matplotlib.pyplot as plt
from cmath import polar
from numpy.linalg import pinv


class twodimspec():
    #Could be change to Rephasing/Non-rephasing naming for better clarity.
    viable_diagrams = ["SE", "SE2", "GSB", "GSB2", "ESA", "ESA2"]

    def __init__(self, propagator, rho_0, w1s=[0], t2s=[0], w3s=[0], color_in = (0,1), color_out=(0,1), tau=10**9, vec="row", color_esa=(0,2), diagrams=[]) -> None:
        """
        color_in, color_out : tuples containg information on which transitions we probe with 1st and 2nd pulses
        for color_in and 3rd and 3th pulses for color_out.
        w1, w3: assume that input is in cmm^-1
        t2 : assume input is in ps
        """
        self.w1_axis = w1s
        self.t2_axis = t2s
        self.w3_axis = w3s
        for diagram in diagrams:
            assert diagram in self.viable_diagrams, ""
        self.diagrams = diagrams
        assert propagator.shape[0] == propagator.shape[1] and int(int(np.sqrt(propagator.shape[0]))**2) == propagator.shape[0], "Incorrect dimensions of propagator. "
        self.propagator = propagator
        # Implement assert check for colors.
        self.color_in = color_in
        self.color_out = color_out
        self.color_esa = color_esa
        self.dim_L = propagator.shape[0]
        self.dim_H = int(np.sqrt(self.dim_L))
        
        self.tau = tau
        self.vec = vec
        

        assert np.all(np.linalg.eigvals(rho_0) > -10e-8), "rho_0 is not positive semi-definite"
        assert round(Decimal(np.trace(rho_0)), 8 ) == round(Decimal(1.0), 8 ), "rho_0 does not have "
        assert rho_0.shape[0] == self.dim_H and rho_0.shape[1] == self.dim_H, "Incorrect density matrix dimension"
        if self.vec == "col":
            self.rho_0 = rho_0.flatten("F")
        elif self.vec == "row":
            self.rho_0 = rho_0.flatten("C")
        else:
            raise Exception("Incorrect assignment of vectorization for density matrix")
        assert vec=="row" or vec=="col", """Incorrect vectorization type. 
    Valid inputs are: 'row' for rowwise and 'col' for columnwise """


        #### INITIALIZE TRANSITIONS ####
        def make_u_op(self, coord):
            x = np.zeros((self.dim_H, self.dim_H))
            x[coord] = 1
            return x
        
        self._u_ge = make_u_op(self, self.color_in)
        self._u_eg = (copy.deepcopy(self._u_ge)).T

        self._u_ge2 = make_u_op(self, self.color_out)
        self._u_eg2 = (copy.deepcopy(self._u_ge2)).T

        if self.vec=="row":

            self._u_eg_left = np.kron(self._u_eg, np.eye(self.dim_H))
            self._u_ge_left = np.kron(self._u_ge, np.eye(self.dim_H))

            self._u_eg_right = np.kron(np.eye(self.dim_H), self._u_eg.T)
            self._u_ge_right = np.kron(np.eye(self.dim_H), self._u_ge.T)
        
            self._u_eg2_left = np.kron(self._u_eg2, np.eye(self.dim_H))
            self._u_ge2_left = np.kron(self._u_ge2, np.eye(self.dim_H))

            self._u_eg2_right = np.kron(np.eye(self.dim_H), self._u_eg2.T)
            self._u_ge2_right = np.kron(np.eye(self.dim_H), self._u_ge2.T)
            
        elif self.vec == "col":

            self._u_eg_left = np.kron(np.eye(self.dim_H), self._u_eg)
            self._u_ge_left = np.kron(np.eye(self.dim_H), self._u_ge)

            self._u_eg_right = np.kron(self._u_eg.T, np.eye(self.dim_H))
            self._u_ge_right = np.kron(self._u_ge.T, np.eye(self.dim_H))
        
            self._u_eg2_left = np.kron(np.eye(self.dim_H),self._u_eg2)
            self._u_ge2_left = np.kron(np.eye(self.dim_H), self._u_ge2)

            self._u_eg2_right = np.kron(self._u_eg2.T, np.eye(self.dim_H))
            self._u_ge2_right = np.kron(self._u_ge2.T, np.eye(self.dim_H))

        if self.dim_H>2:
            self._u_ef = make_u_op(self, color_esa)
            self._u_fe = (copy.deepcopy(self._u_ef)).T

            if self.vec== "row":
                self._u_fe_left = np.kron(self._u_fe, np.eye( self.dim_H))
                self._u_ef_left = np.kron(self._u_ef,np.eye( self.dim_H))

                self._u_fe_right = np.kron(np.eye(self.dim_H), self._u_fe.T )
                self._u_ef_right = np.kron(np.eye(self.dim_H), self._u_ef.T )

            elif self.vec == "col":
                self._u_fe_left = np.kron(np.eye(self.dim_H), self._u_fe)
                self._u_ef_left = np.kron(np.eye(self.dim_H), self._u_ef)

                self._u_fe_right = np.kron(self._u_fe.T, np.eye(self.dim_H))
                self._u_ef_right = np.kron(self._u_ef.T, np.eye(self.dim_H))

        if self.vec == "col":
            self.left_vec = np.conjugate(self._u_eg2.flatten("F")).T
        elif self.vec == "row":
            self.left_vec = np.conjugate(self._u_eg2.flatten("C")).T
        else:
            raise Exception("Incorrect vectorization type in SE_diagram")

    def G_pm_propagate(self, w, sign=1):
        """
        Unit change for w due to t2 plotting in picosecond time unit.
        """
        _w = w* 100 * Lind2DNN.C * 10**-12
        
        A = pinv(sign*1j*_w*np.eye(self.dim_L) + self.propagator)
        B = (LA.expm(sign*1j*_w*np.eye(self.dim_L)+(self.propagator*self.tau)) - np.eye(self.dim_L))
        return np.dot(A, B)
    
    def SE_diagram(self, w1, ta2, w3):
        """
        Stimulated emission diagram contribution to the signal.
        R1 : non-rephasing
        """
        return np.dot(self.left_vec, (self.G_pm_propagate(w3) @ self._u_eg2_right @ LA.expm(self.propagator * ta2) @ self._u_ge_right @ self.G_pm_propagate(w1) @ self._u_eg_left @ self.rho_0))#dens))

    def GSB_diagram(self, w1, ta2, w3):
        """
        Bleach process diagram contribution to the signal.
        R4 : non-rephasing
        """
        return np.dot(self.left_vec, (self.G_pm_propagate(w3) @ self._u_eg2_left @ LA.expm(self.propagator * ta2) @ self._u_ge_left @ self.G_pm_propagate(w1) @ self._u_eg_left @ self.rho_0))#dens))

    def SE2_diagram(self, w1, ta2, w3):
        """
        Stimulated emission diagram contribution to the signal.
        R2 : rephasing
        """
        return np.dot(self.left_vec, (self.G_pm_propagate(w3) @ self._u_eg2_right @ LA.expm(self.propagator * ta2) @ self._u_eg_left @ self.G_pm_propagate(w1, sign=-1) @ self._u_ge_right @ self.rho_0))#ens))

    def GSB2_diagram(self,w1, ta2, w3):
        """
        Bleach process diagram contribution to the signal.
        R3 : rephasing
        """
        return np.dot(self.left_vec, (self.G_pm_propagate(w3) @ self._u_eg2_left @ LA.expm(self.propagator * ta2) @ self._u_eg_right @ self.G_pm_propagate(w1, sign=-1) @ self._u_ge_right @ self.rho_0))#dens))

    diagrams_dict = {"SE": SE_diagram, "SE2": SE2_diagram, "GSB": GSB_diagram, "GSB2": GSB2_diagram}

    def generate_spectrum_data_point(self, t2, w1, w3):
        #out = np.zeros((len(self.t2_axis), len(self.w1_axis), len(self.w3_axis))) 
        out = 0
        for diagram in self.diagrams:
            if diagram in self.diagrams:
                out += (self.diagrams_dict)[diagram](self, w1, t2, w3)
        return out

    def generate_spectrum_data(self, save_to_file=False):
        S_t2t1t3 = np.zeros((len(self.t2_axis), len(self.w3_axis), len(self.w1_axis)), dtype=complex)
        times = [prod for prod in product(self.t2_axis, self.w3_axis, self.w1_axis, repeat=1)]
        start = time.time()

        with mp.Pool(processes = 1) as p:
            results = p.starmap(self.generate_spectrum_data_point, times)
            p.close()
            p.join()
        end = time.time()
        print("Time elapsed: ", end-start)

        for ts2 in range(len(self.t2_axis)):
            slice_size = int(len(self.w1_axis)*len(self.w3_axis))
            temp = np.array(results[(ts2*slice_size):(ts2*slice_size+slice_size)])
            S_t2t1t3[ts2] = temp.reshape(len(self.w3_axis),len(self.w1_axis))
        
        if save_to_file:
            name = ""
            np.savetxt(name, S_t2t1t3)

        return S_t2t1t3

    @staticmethod
    def transform_data(data, log=False, outputs="RNR"):
    
        """
        takes a give 2D slice of the data, makes 2D FFT on it and extracts real and imaginary components.
        These are then rescaled
        Output options: 
        RNR: rephasing/non-rephasing
        RI : real/imaginary
        """

        data =  data*(-1)*(1j)**3
        R = np.zeros(data.real.shape)
        I = np.zeros(data.imag.shape)
        A = np.zeros(data.imag.shape)
        P = np.zeros(data.imag.shape)
    
        if log == True:
            for i in range(len(data)):
                for j in range(len(data)):
                    if data.real[i,j] < 0:
                        R[i,j] = -np.log10(-data.real[i,j])
                    else:
                        R[i,j] = np.log10(data.real[i,j])
                    if data.imag[i,j] < 0:
                        I[i,j] = -np.log10(-data.imag[i,j])
                    else:
                        I[i,j] = np.log10(data.imag[i,j])
                
        elif log == False:
            R = data.real
            I = data.imag
            for i in range(len(data)):
                for j in range(len(data)):
                    pol = polar(data[i,j])
                    A[i,j], P[i,j] = pol[0], pol[1]
                
        return R, I, A, P

if __name__ == "__main__":
    h = np.array([[0,-80],[-80, 800]])
    h = np.array( [[0, -40, -80],[-40, 700, -60],[-80, -60, 800]])

    sim = Lind2DNN(h, E_reorgs=[10], vec="row", unit="ps1", eigenbasis=True)
    U = sim.unitary_dynamics()
    rho_0 = sim.initial_density(density_type="ground", dim=h.shape[0])

    OD, OR, DR, RR = sim.build_lindblad_rate_ops(sim.build_ohmic_bath)
    L_D = sim.lindblad_dissipator(OD, DR)
    L_R = sim.lindblad_dissipator(OR, RR)

    Lind_prop = sum([U, L_D, L_R])
    times = [0.0, 3.8, 10]#[50, 100, 200, 500, 1000]#np.linspace(0.0, 10, 5)#[50, 100]#, 200, 500, 1000]#np.linspace(0.0, 10, 5)# np.linspace(0.0, 10, 10)


    n_w1, n_w3 = 100,100

    w1min, w1max = 600,800#500, 1000#750,950#
    
    w3min, w3max = 750,950#500, 1000#600,800#

    #Shows axis correctly but computes them inverted -> Check computation procedure.

    test_case = twodimspec(Lind_prop, rho_0, w1s=np.linspace(w1min, w1max, n_w1), t2s=times, color_in = (0,2), w3s=np.linspace(w3min, w3max,n_w3),diagrams=[ "SE", "SE2","GSB", "GSB2"], vec="row")

    spec_data = test_case.generate_spectrum_data()
    #eig1 = -8.15071340e-03
    #eig2 = -1.12577112e+00
    #print("predicted timescale: ", 1/abs(eig2-eig1))
    max_val = np.max([v for v in map(np.abs, spec_data.flatten())])
    
    for i, values in enumerate(spec_data):
        fig, ax = plt.subplots()

        a,b,c,d = twodimspec.transform_data(values)
        values = c#a,b,d
        X, Y = np.meshgrid(test_case.w1_axis[:], test_case.w3_axis[:])

        c1 = plt.contourf(X, Y, values, 15, cmap='turbo', vmin=0.0, vmax=max_val)
        
        c2 = plt.contour(X, Y, values, 15, cmap='Greys', vmin=0.0, vmax=max_val)
        name = r"$t_{2} = $" + str(round(Decimal(times[i]),1)) + " ps"

        txt = plt.figtext(0.7,0.8 , name ,ha="right", va="top")#, in_layout=True)
        plt.xlabel(r"$ \omega_{1} [cm^{-1}] $")
        plt.ylabel(r"$ \omega_{3} [cm^{-1}] $")

        txt.set_bbox(dict(facecolor='white', alpha=1, boxstyle="round",capstyle="butt", edgecolor="black"))
        plt.colorbar(c1, ax=ax)
        plt_name = "2DES_plot_t2_" + str(times[i]) + ".eps"
        #plt.axline((w1min, w3min),(w1max,w3max), linewidth=2,  color='w', linestyle="--")#(500,500),(1000,1000)

    
        plt.savefig(plt_name, format = 'eps', dpi=300, transparent = True)
        plt.show()
        plt.close()
    """
    times = np.linspace(0, 1000, 1000)
    Lind_prop = np.asarray(Lind_prop)
    #np.savetxt("Lind_prop.csv", Lind_prop, delimiter = ",")

    e1 = 678.93307929 - -10.87752058
    e2 = 831.94444129 - -10.87752058
    eig1 = -8.15071340e-03
    eig2 = -1.12577112e+00
    eig0 = 0.0
    #pred_time = 1/abs(eig2-eig0)
    #print(pred_time)
    amp_vals = []
    
    amp_test = twodimspec(sum([U, L_D, L_R]), rho_0, w1s=[e2], t2s=times, color_in = (0,2), w3s=[e1],diagrams=["SE", "SE2", "GSB", "GSB2"], vec="row")
    spec_data_amp = amp_test.generate_spectrum_data()
    print("spec", spec_data_amp.shape)
    for val in spec_data_amp:
        amp_vals.append(abs(val[0,0]))

    ys = [LA.norm(LA.expm(Lind_prop*t), 2) for t in times]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    name_1 = r"$ \mathbf{L} = \mathbf{L_U} + \mathbf{L_D} + \mathbf{L_R}$"
    name_2 = r"$ S^{(3)}(\omega_{1}, t_{2}, \omega_{3})$"
    plot1  = ax1.plot(times, ys, color="b",ls="dashdot", linewidth=2,  label=name_1)

    #ax1.plot(times, amp_vals, ".",  label="Amps" )
    #plt.hlines(np.amax(amp_vals), xmin=times[0], xmax=times[-1])
    #plt.hlines(np.amax(ys),xmin=times[0], xmax=times[-1], colors="r")
    #plt.vlines(pred_time, ymin=min(ys), ymax=max(ys),  color="y", label=r"$ Timescale \ from \ spec(\mathbf{L})$")
    ax1.set_xlabel(r"$ Time [ps^{-1}]$")
    ax1.set_ylabel(r"$ \Vert e^{\mathbf{L}t}\Vert$")
    plt.grid()
    
    ax2.set_ylabel(r"$ S^{3}(\omega_{1} = $" + str(round(Decimal(e1),0))+ r"$ cm^{-1}$"  + r"$, t_{2}$" + r"$, \omega_{3}=$"+ str(round(Decimal(e2),0)) + r"$ cm^{-1} )$" )
    plot2 = ax2.plot(times, amp_vals, color="y",ls="dotted", linewidth=2,  label=name_2)


    lns = plot1+plot2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    #plt.show()

    #ax2.legend()
    #ax2.legend()
    #plt.savefig("Lnorm_3lvl_longtime.eps", format = 'eps', dpi=300, transparent=True)
    #print("dens 11: ", times[np.argmax(amp_vals)], "L_norm: ", times[np.argmax(ys)] )
    """