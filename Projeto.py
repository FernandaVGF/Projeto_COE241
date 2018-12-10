import math
import matplotlib as mpl
import numpy as np
import pylab
import re
import scipy.stats as stats
import statistics

## 2.1
amostras = []

with open ('Dados-medicos.csv') as arquivo:
    for linha in arquivo:
        linha_2 = re.split ('   | |\n', linha)
        for elemento in linha_2:
            if elemento != '':
                amostras.append (float (elemento))
                
indice = 0

idade, indice_idade = [], 0
peso, indice_peso = [], 1
carga, indice_carga = [], 2
vo2, indice_vo2 = [], 3

for elemento in amostras:
    
    if indice == indice_idade:
        idade.append (elemento)
        indice_idade += 4
        
    elif indice == indice_peso:
        peso.append (elemento)
        indice_peso += 4
        
    elif indice == indice_carga:
        carga.append (elemento)
        indice_carga += 4
    
    elif indice == indice_vo2:
        vo2.append (elemento)
        indice_vo2 += 4
        
    indice += 1


# Histograma
n = len (idade) # Número de amostras
m = 1 + 3.3*(math.log (n, 10)) # Número de intervalos

# mpl.pyplot.hist (idade, bins = int (m), normed = True)
# mpl.pyplot.hist (peso, bins = int (m), normed = True)
# mpl.pyplot.hist (carga, bins = int (m), normed = True)
# mpl.pyplot.hist (vo2, bins = int (m), normed = True)


# Função distribuição empírica
x_idade = np.sort (idade)
x_peso = np.sort (peso)
x_carga = np.sort (carga)
x_vo2 = np.sort (vo2)

y_demp_idade = np.arange (len (x_idade))/float (n)
y_demp_peso = np.arange (len (x_peso))/float (n)
y_demp_carga = np.arange (len (x_carga))/float (n)
y_demp_vo2 = np.arange (len (x_vo2))/float (n)

# mpl.pyplot.plot (x_idade, y_demp_idade)
# mpl.pyplot.plot (x_peso, y_demp_peso)
# mpl.pyplot.plot (x_carga, y_demp_carga)
# mpl.pyplot.plot (x_vo2, y_demp_vo2)



## 2.2
# Média
media_idade = statistics.mean (idade)
media_peso = statistics.mean (peso)
media_carga = statistics.mean (carga)
media_vo2 = statistics.mean (vo2)


# Variância amostral
soma_v_am_idade, soma_v_am_peso, soma_v_am_carga, soma_v_am_vo2 = [0]*4

for elemento in idade:
    soma_v_am_idade += (elemento - media_idade)**2
    
for elemento in peso:
    soma_v_am_peso += (elemento - media_peso)**2
    
for elemento in carga:
    soma_v_am_carga += (elemento - media_carga)**2
    
for elemento in vo2:
    soma_v_am_vo2 += (elemento - media_vo2)**2
    
v_am_idade = soma_v_am_idade/(n-1)
v_am_peso = soma_v_am_peso/(n-1)
v_am_carga = soma_v_am_carga/(n-1)
v_am_vo2 = soma_v_am_vo2/(n-1)


# Boxplot
# mpl.pyplot.boxplot (idade)
# mpl.pyplot.boxplot (peso)
# mpl.pyplot.boxplot (carga)
# mpl.pyplot.boxplot (vo2)



## 2.3
# Exponencial
lbd_idade = len (idade)/sum (idade)
lbd_peso = len (peso)/sum (peso)
lbd_carga = len (carga)/sum (carga)
lbd_vo2 = len (vo2)/sum (vo2)

y_exp_idade = stats.expon.cdf (x_idade, scale = 1/lbd_idade)
y_exp_peso = stats.expon.cdf (x_peso, scale = 1/lbd_peso)
y_exp_carga = stats.expon.cdf (x_carga, scale = 1/lbd_carga)
y_exp_vo2 = stats.expon.cdf (x_vo2, scale = 1/lbd_vo2)

# mpl.pyplot.plot (x_idade, y_exp_idade)
# mpl.pyplot.plot (x_peso, y_exp_peso)
# mpl.pyplot.plot (x_carga, y_exp_carga)
# mpl.pyplot.plot (x_vo2, y_exp_vo2)


# Gaussiana (normal)
media_norm_idade = sum (idade)/n
media_norm_peso = sum (peso)/n
media_norm_carga = sum (carga)/n
media_norm_vo2 = sum (vo2)/n

soma_v_norm_idade, soma_v_norm_peso, soma_v_norm_carga, soma_v_norm_vo2 = [0]*4

for elemento in idade:
    soma_v_norm_idade += (elemento - media_norm_idade)**2
    
for elemento in peso:
    soma_v_norm_peso += (elemento - media_norm_peso)**2
    
for elemento in carga:
    soma_v_norm_carga += (elemento - media_norm_carga)**2
    
for elemento in vo2:
    soma_v_norm_vo2 += (elemento - media_norm_vo2)**2

v_norm_idade, v_norm_peso, v_norm_carga, v_norm_vo2 = soma_v_norm_idade/n, soma_v_norm_peso/n, soma_v_norm_carga/n, soma_v_norm_vo2/n

y_norm_idade = stats.norm.cdf (x_idade, loc = media_norm_idade, scale = math.sqrt (v_norm_idade))
y_norm_peso = stats.norm.cdf (x_peso, loc = media_norm_peso, scale = math.sqrt (v_norm_peso))
y_norm_carga = stats.norm.cdf (x_carga, loc = media_norm_carga, scale = math.sqrt (v_norm_carga))
y_norm_vo2 = stats.norm.cdf (x_vo2, loc = media_norm_vo2, scale = math.sqrt (v_norm_vo2))

# mpl.pyplot.plot (x_idade, y_norm_idade)
# mpl.pyplot.plot (x_peso, y_norm_peso)
# mpl.pyplot.plot (x_carga, y_norm_carga)
# mpl.pyplot.plot (x_vo2, y_norm_vo2)


# Lognormal
soma_media_ln_idade, soma_media_ln_peso, soma_media_ln_carga, soma_media_ln_vo2, soma_v_ln_idade, soma_v_ln_peso, soma_v_ln_carga, soma_v_ln_vo2 = [0]*8

for elemento in idade:
    soma_media_ln_idade += math.log (elemento)
    
for elemento in peso:
    soma_media_ln_peso += math.log (elemento)
    
for elemento in carga:
    soma_media_ln_carga += math.log (elemento)
    
for elemento in vo2:
    soma_media_ln_vo2 += math.log (elemento)
    
media_ln_idade = soma_media_ln_idade/n
media_ln_peso = soma_media_ln_peso/n
media_ln_carga = soma_media_ln_carga/n
media_ln_vo2 = soma_media_ln_vo2/n

for elemento in idade:
    soma_v_ln_idade += (math.log (elemento) - media_ln_idade)**2
    
for elemento in peso:
    soma_v_ln_peso += (math.log (elemento) - media_ln_peso)**2
    
for elemento in carga:
    soma_v_ln_carga += (math.log (elemento) - media_ln_carga)**2
    
for elemento in vo2:
    soma_v_ln_vo2 += (math.log (elemento) - media_ln_vo2)**2
    
v_ln_idade = soma_v_ln_idade/n
v_ln_peso = soma_v_ln_peso/n
v_ln_carga = soma_v_ln_carga/n
v_ln_vo2 = soma_v_ln_vo2/n

y_ln_idade = stats.lognorm.cdf (x_idade, s = math.sqrt (v_ln_idade), scale = math.exp (media_ln_idade))
y_ln_peso = stats.lognorm.cdf (x_peso, s = math.sqrt (v_ln_peso), scale = math.exp (media_ln_peso))
y_ln_carga = stats.lognorm.cdf (x_carga, s = math.sqrt (v_ln_carga), scale = math.exp (media_ln_carga))
y_ln_vo2 = stats.lognorm.cdf (x_vo2, s = math.sqrt (v_ln_vo2), scale = math.exp (media_ln_vo2))

# mpl.pyplot.plot (x_idade, y_ln_idade)
# mpl.pyplot.plot (x_peso, y_ln_peso)
# mpl.pyplot.plot (x_carga, y_ln_carga)
# mpl.pyplot.plot (x_vo2, y_ln_vo2)


# Weibull
param_weibull_idade = stats.weibull_min.fit (idade)
param_weibull_peso = stats.weibull_min.fit (peso)
param_weibull_carga = stats.weibull_min.fit (carga)
param_weibull_vo2 = stats.weibull_min.fit (vo2)

y_weib_idade = stats.weibull_min.cdf (x_idade, param_weibull_idade [0], param_weibull_idade [1], param_weibull_idade [2])
y_weib_peso = stats.weibull_min.cdf (x_peso, param_weibull_peso [0], param_weibull_peso [1], param_weibull_peso [2])
y_weib_carga = stats.weibull_min.cdf (x_carga, param_weibull_carga [0], param_weibull_carga [1], param_weibull_carga [2])
y_weib_vo2 = stats.weibull_min.cdf (x_vo2, param_weibull_vo2 [0], param_weibull_vo2 [1], param_weibull_vo2 [2])

# mpl.pyplot.plot (x_idade, y_weib_idade)
# mpl.pyplot.plot (x_peso, y_weib_peso)
# mpl.pyplot.plot (x_carga, y_weib_carga)
# mpl.pyplot.plot (x_vo2, y_weib_vo2)



## 2.4
# stats.probplot (idade, dist = stats.expon (scale = lbd_idade), plot = pylab)
# stats.probplot (idade, dist = stats.norm (loc = media_norm_idade, scale = math.sqrt (v_norm_idade)), plot = pylab)
# stats.probplot (idade, dist = stats.lognorm (s = math.sqrt (v_ln_idade), scale = math.exp (media_ln_peso)), plot = pylab)
# stats.probplot (idade, dist = stats.weibull_min (param_weibull_idade [0], param_weibull_idade [1], param_weibull_idade [2]), plot = pylab)

# stats.probplot (peso, dist = stats.expon (scale = lbd_peso), plot = pylab)
# stats.probplot (peso, dist = stats.norm (loc = media_norm_peso, scale = math.sqrt (v_norm_peso)), plot = pylab)
# stats.probplot (peso, dist = stats.lognorm (s = math.sqrt (v_ln_peso), scale = math.exp (media_ln_peso)), plot = pylab)
# stats.probplot (peso, dist = stats.weibull_min (param_weibull_peso [0], param_weibull_peso [1], param_weibull_peso [2]), plot = pylab)

# stats.probplot (carga, dist = stats.expon (scale = lbd_carga), plot = pylab)
# stats.probplot (carga, dist = stats.norm (loc = media_norm_carga, scale = math.sqrt (v_norm_carga)), plot = pylab)
# stats.probplot (carga, dist = stats.lognorm (s = math.sqrt (v_ln_carga), scale = math.exp (media_ln_carga)), plot = pylab)
# stats.probplot (carga, dist = stats.weibull_min (param_weibull_carga [0], param_weibull_carga [1], param_weibull_carga [2]), plot = pylab)

# stats.probplot (vo2, dist = stats.expon (scale = lbd_vo2), plot = pylab)
# stats.probplot (vo2, dist = stats.norm (loc = media_norm_vo2, scale = math.sqrt (v_norm_vo2)), plot = pylab)
# stats.probplot (vo2, dist = stats.lognorm (s = math.sqrt (v_ln_vo2), scale = math.exp (media_ln_vo2)), plot = pylab)
# stats.probplot (vo2, dist = stats.weibull_min (param_weibull_vo2 [0], param_weibull_vo2 [1], param_weibull_vo2 [2]), plot = pylab)



## 2.5
'''H0 (True): para todo x, F_empirica = F_parametrizada
H1 (False): existe um x tal que F_empirica =! F_parametrizada'''

d_n_alfa = 1.3581/math.sqrt (n) # n > 50, alfa = 0.05

# Idade
d_idade_exp = abs (np.amax (y_exp_idade - y_demp_idade))
d_idade_norm = abs (np.amax (y_norm_idade - y_demp_idade))
d_idade_ln = abs (np.amax (y_ln_idade - y_demp_idade))
d_idade_weib = abs (np.amax (y_weib_idade - y_demp_idade))

# print ("Idade (exponencial): " + str (d_idade_exp <= d_n_alfa))
# print ("Idade (gaussiana): " + str (d_idade_norm <= d_n_alfa))
# print ("Idade (lognormal): " + str (d_idade_ln <= d_n_alfa))
# print ("Idade (weibull): " + str (d_idade_weib <= d_n_alfa))


# Peso
d_peso_exp = abs (np.amax (y_exp_peso - y_demp_peso))
d_peso_norm = abs (np.amax (y_norm_peso - y_demp_peso))
d_peso_ln = abs (np.amax (y_ln_peso - y_demp_peso))
d_peso_weib = abs (np.amax (y_weib_peso - y_demp_peso))

# print ("Peso (exponencial): " + str (d_peso_exp <= d_n_alfa))
# print ("Peso (gaussiana): " + str (d_peso_norm <= d_n_alfa))
# print ("Peso (lognormal): " + str (d_peso_ln <= d_n_alfa))
# print ("Peso (weibull): " + str (d_peso_weib <= d_n_alfa))


# Carga final
d_carga_exp = abs (np.amax (y_exp_carga - y_demp_carga))
d_carga_norm = abs (np.amax (y_norm_carga - y_demp_carga))
d_carga_ln = abs (np.amax (y_ln_carga - y_demp_carga))
d_carga_weib = abs (np.amax (y_weib_carga - y_demp_carga))

# print ("Carga final (exponencial): " + str (d_carga_exp <= d_n_alfa))
# print ("Carga final (gaussiana): " + str (d_carga_norm <= d_n_alfa))
# print ("Carga final (lognormal): " + str (d_carga_ln <= d_n_alfa))
# print ("Carga final (weibull): " + str (d_carga_weib <= d_n_alfa))


# VO2 máximo
d_vo2_exp = abs (np.amax (y_exp_vo2 - y_demp_vo2))
d_vo2_norm = abs (np.amax (y_norm_vo2 - y_demp_vo2))
d_vo2_ln = abs (np.amax (y_ln_vo2 - y_demp_vo2))
d_vo2_weib = abs (np.amax (y_weib_vo2 - y_demp_vo2))

# print ("VO2 máximo (exponencial): " + str (d_vo2_exp <= d_n_alfa))
# print ("VO2 máximo (gaussiana): " + str (d_vo2_norm <= d_n_alfa))
# print ("VO2 máximo (lognormal): " + str (d_vo2_ln <= d_n_alfa))
# print ("VO2 máximo (weibull): " + str (d_vo2_weib <= d_n_alfa))



## 2.6
# Coeficiente de correlação amostral (r)
r_idade = (stats.pearsonr (idade, vo2)) [0]
r_peso = (stats.pearsonr (peso, vo2)) [0]
r_carga = (stats.pearsonr (carga, vo2)) [0]


# Scatter plot
# mpl.pyplot.scatter (idade, vo2) # Idade (x) x VO2 máximo (y)
# mpl.pyplot.scatter (peso, vo2) # Peso (x) x VO2 máximo (y)
# mpl.pyplot.scatter (carga, vo2) # Carga final (x) x VO2 máximo (y)


# Modelo de regressão (y = a + bx)
a = np.polyfit (carga, vo2, 1) [1]
b = np.polyfit (carga, vo2, 1) [0]

# mpl.pyplot.plot (x_carga, a + b*x_carga)



## 2.7
'''h1 = (30 <=) carga < 230
h2 = 230 <= carga (<= 432)'''

# Tabela 1: VO2 < 35 (D1)
'''
hip.      prior = P (h)      likelihood = P (D1|h)      numerador de Bayes = P (h) * P (D1|h)      posterior = P (h|D1)
h1        prob_h1            likel_tab1_h1              a = prob_h1 * likel_tab1_h1                a * 1/T1
h2        prob_h2            likel_tab1_h2              b = prob_h2 * likel_tab1_h2                b * 1/T1
          -> 1                                         -> P (D1) = T1 = a + b                      -> 1
'''
soma_h1, soma_h2, casos_tab1_h1, casos_tab1_h2, casos_tab2_h1, casos_tab2_h2 = [0]*6

for indice in range (n):
    if carga [indice] < 230:
        soma_h1 += 1
        
        if vo2 [indice] < 35:
            casos_tab1_h1 += 1
            
        else:
            casos_tab2_h1 += 1
        
    else:
        soma_h2 += 1
        
        if vo2 [indice] < 35:
            casos_tab1_h2 += 1
            
        else:
            casos_tab2_h2 += 1 

prob_h1 = soma_h1/n
prob_h2 = soma_h2/n

likel_tab1_h1 = casos_tab1_h1/soma_h1
likel_tab1_h2 = casos_tab1_h2/soma_h2

a = prob_h1 * likel_tab1_h1
b = prob_h2 * likel_tab1_h2
p_d1 = a + b

posterior_tab1_h1 = a/p_d1
posterior_tab1_h2 = b/p_d1


# Tabela 2: VO2 >= 35 (D2)
'''
hip.      prior = P (h)      likelihood = P (D2|h)      numerador de Bayes = P (h) * P (D2|h)      posterior = P (h|D2)
h1        prob_h1            likel_tab2_h1              c = prob_h1 * likel_tab2_h1                c * 1/T2
h2        prob_h2            likel_tab2_h2              d = prob_h2 * likel_tab2_h2                d * 1/T2
          -> 1                                          -> P (D2) = T2 = c + d                     -> 1
'''
likel_tab2_h1 = casos_tab2_h1/soma_h1
likel_tab2_h2 = casos_tab2_h2/soma_h2

c = prob_h1 * likel_tab2_h1
d = prob_h2 * likel_tab2_h2
p_d2 = c + d

posterior_tab2_h1 = c/p_d2
posterior_tab2_h2 = d/p_d2


# Tabela 3: VO2 >= 35 (D2) depois de VO2 < 35 (D1)
'''
hip.      prior        likelihood 1       numerador de Bayes 1             posterior 1      likelihood 2 = *      numerador de Bayes 2 = **
h1        prob_h1      likel_tab1_h1      a = prob_h1 * likel_tab1_h1      a * 1/T1         likel_tab2_h1         e = a * 1/T1 * likel_tab2_h1
h2        prob_h2      likel_tab1_h2      b = prob_h1 * likel_tab1_h2      b * 1/T1         likel_tab2_h2         f = b * 1/T1 * likel_tab2_h2
          -> 1                            -> P (D1) = T1 = a + b           -> 1                            

* = P (D2|h, D1)
** = P (D2|h, D1) * P (h|D1) = P (D2|D1) = e + f
'''
e = posterior_tab1_h1 * likel_tab2_h1
f = posterior_tab1_h2 * likel_tab2_h2

p_d2_d1 = e + f