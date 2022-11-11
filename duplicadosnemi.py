# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:02:23 2022

              EXAMEN PARCIAL DE FENÓMENOS DE TRANSPORTE
Cátedra ...... Fenomenos de Trasporte
Profesor ..... Edgardo A. Serafin
Examen Parcial .......TEMA 1 · difusión 1D (aleta de enfriamiento)
Autor: TP Federico CACERES

Considere el problema de conduccion de calor en una aleta de enfriamiento segun
se muestra en la Figura 1 y analicelo como un problema cuasi-unidimensional en
el eje x.
El dominio del problema es una aleta circular conica truncada con un radio r(x)=ax +b
donde a y b son constantes.El largo de la aleta el L=10cm,con r1=1cm y r2=0.5cm.
La aleta pierde calor a traves de la superficie lateral envolvente hacia una atmosfera
que se encuentra a 27°C con un coeficiente de transferencia h = 128 W/m^2K. La
base de la aleta (en x = 0) se encuentra adosada a una pared cuya temperatura es TA = 127 °C
y la punta ( en x = L ) esta adecuadamente aislada. Todas las propiedades pueden
considerars constantes. La conductividad termica del material de la aleta, fabricado
en una aleacion de bronce, puede asumirse como k = 250 W/mK.
"""
"""
"""
import numpy as np
import matplotlib.pyplot as plt
    
    # ------------------------------- PARÁMETROS de ENTRADA------------------------
for nx, color in zip([80, 20, 5], ['r', 'g', 'b']):
    xLen = 0.1                 # longitud del dominio en dirección x en [m]
    #nx   = 5 #20 #80           # número de volumenes en dirección x
    Tamb = 27.0                # temperatura ambiente
    TA   = 127.0               # temperatura en la pared x = 0
    # Funcion lineal con "r" variable = r(x)= a*x+b
    a=-0.05
    b=0.01
    h= 128                       # W/m^2K
    R1=0.01                      #radio en [m]
    R2=0.005
    dx = xLen/nx
    k= 250                       # W/mk
    xs = np.linspace(0,xLen,nx+1)
    xp = np.linspace(dx/2,xLen-dx/2,nx)
    rs = (a*xs)+b
    rp = (a*xp)+b                  # radio en los centroides
    Ap= np.pi*rp**2                # area
    Ac= np.pi*rs**2
    #rp1= (rs[0:nx]+rs[1:nx+1])/2   # radio en los centroides
    Perc= 2*np.pi*rp               # perimetro en los centroides
    n2=h*Perc/k               # número constante: hP/(kA)
    k = 1
    # ---------------------------------- DATOS de CONTROL--------------------------
    iterMax=3000               # máximo número de iteraciones
    toler=10**(-2)             # tolerancia (error) para la convergencia - 0,01
    # -----------------------------------------------------------------------------
    #-----------------------------------GEOMETRÍA - MALLADO)-----------------------
    # --- coordenada para centroides
    dy = 1.                    # tamaño del CV en y
    esp  = 1.                  # espesor
    volp = dx*dy*esp           # volumen
    D = Ac/dx                #coeficiente D
    # ---------------------------COEFICIENTES VECINOS------------------------------
    aw = np.copy(D[0:nx]);       ae = np.copy(D[1:nx+1]);
    # ---       
    aw[0] = 0.0;   ae[nx-1] = 0.0
    bp = np.zeros(nx)
    bc = np.zeros(nx)
    # -------------------------- CONDICIONES de CONTORNO---------------------------
    bc[0]    = bc[0] + 2*D[0]*TA            # cc W (DIRICHLET: T=TA)
    bp[0]    = bp[0] - 2*D[0]
    bc[nx-1] = bc[nx-1] + 0                 # cc E (NEUMANN: adiabático)
    bp[nx-1] = bp[nx-1] - 0
    # ------------------------------ FUENTES y aP----------------------------------
    sp = -n2*dx*np.ones(nx)
    sc = n2*dx*Tamb*np.ones(nx)
    ap = aw + ae - sp - bp
    ##### CÁLCULO de PHI por GAUSS-SEIDEL #########################################
    # ------------ EXPANSIÓN (ny+2,nx+2)-------------------------------------------
    aw0=np.pad(aw,(1));     ae0=np.pad(ae,(1))
    bc0=np.pad(bc,(1));     bp0=np.pad(bp,(1)) 
    sc0=np.pad(sc,(1));     sp0=np.pad(sp,(1))
    # ---
    ap0=np.pad(ap,(1,1),'constant',constant_values=(0,0))
    ##### CÁLCULO de PHI por MÉTODO ITERATIVO #####################################
    # --- semilla y residuos
    TT=100*np.ones(nx)                       # semilla
    T0=np.pad(TT,(1))                        # expansión (ny+2,nx+2)
    s=0                                      # contador en cero
    # ---------------------------------------------------------------- GAUSS-SEIDEL
    for k in range (0,iterMax):
        Tprev=np.copy(T0)
        for i in range (1,nx+1):             # fijo las filas (y)
            # --- loop phi_P = [sumatoria(a_nb*phi_nb) + b]/an_P
            T0[i] = (aw0[i]*T0[i-1] + ae0[i]*T0[i+1] + sc0[i] + bc0[i]) / ap0[i]
        # ----------------------------------------------- CONVERGENCIA y TOLERANCIA
        d = np.linalg.norm(Tprev-T0)         # cálculo de norma2 T previa vs actual
        s=s+1                                # sumador de iteraciones
        if d < toler:                        # control condición de convergencia
            break
    # ---------------------------------------- CONTRACCIÓN (ny,nx)
    T=np.ones((nx))
    T[0:nx]=T0[1:nx+1]
    ##### VISUALIZACIÓN 2D ########################################################
    xx1 = np.linspace(0, xLen, 55); n= np.sqrt((h*(2*np.pi*R1))/(250*(np.pi*R1**2)))
    yy1 = np.cosh(n*(xLen-xx1))/np.cosh(n*xLen)*(TA-Tamb)+Tamb
    xx2 = np.linspace(0, xLen, 55); n= np.sqrt((h*(2*np.pi*R2))/(250*(np.pi*R2**2)))
    yy2 = np.cosh(n*(xLen-xx1))/np.cosh(n*xLen)*(TA-Tamb)+Tamb
    Tmax = np.max(T)                         # temperatura máxima
    Tmin = np.min(T)                         # temperatura mínima
    xp1 = np.insert(xp,0,0.0)                # centroides + cara w
    T1 = np.insert(T,0,TA)                   # T + cara w
    # ------------------------------------------------------ VISUALIZACIÓN PHI vs X
    SERA1 = plt.figure(1)
    plt.plot(xp1, T1, ':k', marker='o', markerfacecolor=color, markersize=5, label=('FVM '+str(nx)))
plt.title('EXAMEN PARCIAL DE FENÓMENOS DE TRANSPORTE ', fontsize=12)
#plt.xlim(0.0,1.1*xLen) 
plt.plot(xx1,yy1,'-b', lw=2.0,label='analitica; 1.0 cm')
plt.plot(xx2,yy2,'-r', lw=2.0,label='analitica: 0.5 cm')
plt.ylim(65,130)
plt.grid(True)
plt.xlabel('Posición x  [m]')
plt.ylabel('Temperatura  [$^\circ$C]')
plt.legend(frameon=True, fontsize=10)
plt.show()
# --- salida en pantalla
print('\n---------------- --- EXAMEN PARCIAL DE FENÓMENOS DE TRANSPORTE ---------')
print('                   cantidad de CVs = {} '.format(nx))
print('                                dx = {:1.3f} '.format(dx))
print('                          T máxima = {:1.2f} '.format(Tmax))
print('                          T mínima = {:1.2f} '.format(Tmin))