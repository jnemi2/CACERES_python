# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 07:30:14 2022

@author: fede_
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor
@author: aldo

###############################################################################
#########################                             #########################
#########################      SP01 · PROBLEMA 2      #########################
#########################           phantom           #########################
###############################################################################

Resuelve el problema de difusón en dos dimensiones por FVM con mallado homoge-
neo y resolución por método iterativo (Gauss-Seidel) en un dominio de sección
rectangular en coordenadas cartesianas con/sin generación y con toda la varie-
dad posible de cndiciones de contorno.
-------------------------------------------------------------------------------
Cátedra ...... Introducción a la Dinámica de Fluidos Computacional
Profesor ..... Edgardo A. Serafin
SP 01 ........ Problema 2 · difusión 1D (aleta de enfriamiento)
Facultad de Ingeniería del Ejército / Universidad de la Defensa Nacional
Ciudad Autónoma de Buenos Aires, 2021
(c) 2021 - EAS / SiCaNLab-FIE
---------------------------------------------------------- PARÁMETROS de SALIDA
T = campo de temperatura resultante en el dominio de estudio
x = coordenadas de los centroides en x (cartesianas)
graf 1 = gráfico de temperatura vs posición en distintos sectores del dominio
###############################################################################
###############################################################################
###############################################################################
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
t0 = time()         # inicio de contador de tiempo
# ------------------------------------------------------- PARÁMETROS de ENTRADA
xLen = 1.0                 # longitud del dominio en dirección x
nx   = 5                   # número de volumenes en dirección x
n2   = 25.0                # número constante: hP/(kA)
Tamb = 20.0                # temperatura ambiente
TA   = 100.0               # temperatura en extremo x = 0
k    = 1.                  # conductividad térmica
# -------------------------------------------------------- PARÁMETROS DERIVADOS
dx = xLen/nx
# ------------------------------------------------------------ DATOS de CONTROL
iterMax=3000               # máximo número de iteraciones
toler=10**(-2)             # tolerancia (error) para la convergencia

###############################################################################
##### DISCRETIZACIÓN FVM 2D (GEOMETRÍA - MALLADO) #############################
###############################################################################
# --- coordenada para centroides
xp = np.linspace(dx/2,xLen-dx/2,nx)      
dx = xLen/nx               # tamaño del Cv en x
dy = 1.                    # tamaño del CV en y
esp  = 1.                  # espesor
volp = dx*dy*esp           # volumen

###############################################################################
##### DISCRETIZACIÓN FVM 2D (ECUACIÓN de CONSERVACIÓN) ########################
###############################################################################
D = k*dy/dx;               # coeficientes D
# -------------------------------------------------------- COEFICIENTES VECINOS
aw = D*np.ones(nx);       ae = D*np.ones(nx);
# ---       
aw[0] = 0.0;   ae[nx-1] = 0.0
bp = np.zeros(nx)
bc = np.zeros(nx)
# ----------------------------------------------------- CONDICIONES de CONTORNO
bc[0]    = bc[0] + 2*D*TA                # cc W (DIRICHLET: T=TA)
bp[0]    = bp[0] - 2*D
bc[nx-1] = bc[nx-1] + 0                  # cc E (NEUMANN: adiabático)
bp[nx-1] = bp[nx-1] - 0
# ---------------------------------------------------------------- FUENTES y aP
sp = -n2*dx*np.ones(nx)
sc = n2*dx*Tamb*np.ones(nx)
ap = aw + ae - sp - bp
# ------------------------------------ SOLUCIÓN ANALÍTICA: con y sin generación
xx = np.linspace(0,xLen,num=50);          n = np.sqrt(n2)
yy = np.cosh(n*(xLen-xx))/np.cosh(n*xLen)*(TA-Tamb) + Tamb

###############################################################################
##### CÁLCULO de PHI por GAUSS-SEIDEL #########################################
###############################################################################
# ------------------------------------------------------- EXPANSIÓN (ny+2,nx+2)
aw0=np.pad(aw,(1));     ae0=np.pad(ae,(1))
bc0=np.pad(bc,(1));     bp0=np.pad(bp,(1)) 
sc0=np.pad(sc,(1));     sp0=np.pad(sp,(1))
# ---
ap0=np.pad(ap,(1,1),'constant',constant_values=(0,0))

###############################################################################
##### CÁLCULO de PHI por MÉTODO ITERATIVO #####################################
###############################################################################
# --- semilla y residuos
TT=50*np.ones(nx)                        # semilla
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
# --------------------------------------------------------- CONTRACCIÓN (ny,nx)
T=np.ones((nx))
T[0:nx]=T0[1:nx+1]

###############################################################################
##### VISUALIZACIÓN 2D ########################################################
###############################################################################
Tmax = np.max(T)                         # temperatura máxima
Tmin = np.min(T)                         # temperatura mínima
xp1 = np.insert(xp,0,0.0)                # centroides + cara w
T1 = np.insert(T,0,TA)                   # T + cara w

# ------------------------------------------------------ VISUALIZACIÓN PHI vs X
SERA1 = plt.figure(1)
plt.plot(xp1,T1,':k',marker='o',markerfacecolor='r',markersize=9)
plt.plot(xx,yy,'-b',linewidth=2.0)
plt.title('SP01 · Problema 2 (aleta)', fontsize=14)
plt.xlim(-0.05,1.05*xLen) 
plt.ylim(0.6*Tamb,1.1*TA)
plt.grid(True)
plt.xlabel('posición x  [m]')
plt.ylabel('temperatura  [$^\circ$C]')
plt.legend(['FVM','analitica'], frameon=True, fontsize=8)
plt.show()

# ----------------------------------------------------------- DATOS en PANTALLA
t1 = time()                              # final del contador de tiempo
tiempo = t1-t0                           # tiempo de simulación
# --- salida en pantalla
print('\n---------------- --- SP01 · Problema 2 (aleta) ----------- ---------')
print('                   cantidad de CVs = {} '.format(nx))
print('                                dx = {:1.3f} '.format(dx))
print('                          T máxima = {:1.2f} '.format(Tmax))
print('                          T mínima = {:1.2f} '.format(Tmin))
print('           cantidad de iteraciones = {} '.format(s))
print('                      tiempo total = {:1.1f} sec'.format(tiempo))

###############################################################################
###################################   FIN   ###################################
###############################################################################