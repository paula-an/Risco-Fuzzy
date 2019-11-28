import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

#
#
# Função para traçar fuzzy sets
def plotmf(universe, names, mf, xticks, xlabel):
    plt.figure()
    for name in names:
        plt.plot(universe, mf[name].mf, label=name)
    plt.legend() 
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('Pertinência')
    plt.xticks(xticks)
    plt.show()
    
#
# Função para fuzzyficação
def myfuzzification(universe, names, mf, val_in):
    val_fuzz = {}
    for name in names:
        val_fuzz[name] = fuzz.interp_membership(universe, mf[name].mf, val_in)
    return val_fuzz

#
#
# Variáveis fuzzy de entrada
#
# Universo das variáveis [0 a 1]
uni = np.arange(0, 100.5, 0.5)
xticks = np.arange(0, 110, 10)

#
# in_inv
nomes_inv = ['B', 'M', 'A']
in_inv = ctrl.Antecedent(uni, 'in_inv')
in_inv['B'] = fuzz.trimf(uni, [0, 0, 60])  # Orçamento baixo
in_inv['M'] = fuzz.trimf(uni, [40, 70, 100])  # Orçamento médio
in_inv['A'] = fuzz.trapmf(uni, [60, 90, 100, 100])  # Orçamento adequado
plotmf(uni, nomes_inv, in_inv, xticks, 'Orçamento (%)')


#
# in_pes
nomes_pes = ['B', 'A']
in_pes = ctrl.Antecedent(uni, 'in_pes')
in_pes['B'] = fuzz.trimf(uni, [0, 0, 100])  # Orçamento baixo
in_pes['A'] = fuzz.trimf(uni, [0, 100, 100])  # Orçamento adequado
plotmf(uni, nomes_pes, in_pes, xticks, 'Pessoal (%)')

#
# out_risk
nomes_risk = ['B', 'M', 'A']
out_risk = ctrl.Antecedent(uni, 'out_risk')
out_risk['B'] = fuzz.trapmf(uni, [0, 0, 30, 50])  # Risco baixo
out_risk['M'] = fuzz.trimf(uni, [20, 50, 80])  # Risco médio
out_risk['A'] = fuzz.trapmf(uni, [50, 70, 100, 100])  # Risco baixo
plotmf(uni, nomes_risk, out_risk, xticks, 'Risco (%)')
#
# Entradas
val_inv = 50
val_pes = 50

#
# Fuzzificação
# Orçamento
fuzz_inv = myfuzzification(uni, nomes_inv, in_inv, val_inv)
# Pessoal
fuzz_pes = myfuzzification(uni, nomes_pes, in_pes, val_pes)
#
#
# Inferência (regras)
fuzz_risk = {}
#
# Regra 1 - Se orçamento B E pessoal B, então risco A
# usa np.fmax para OU e np.fmin para E
rule1 = np.fmin(fuzz_inv['B'], fuzz_pes['B'])
fuzz_risk['A'] = np.fmin(rule1, out_risk['A'].mf)
#
# Regra 2 - Se orçamento M E pessoal A, então risco M
# usa np.fmax para OU e np.fmin para E
rule2 = np.fmin(fuzz_inv['M'], fuzz_pes['A'])
fuzz_risk['M'] = np.fmin(rule2, out_risk['M'].mf)
#
# Regra 3 - Se orçamento A, então risco B
# usa np.fmax para OU e np.fmin para E
fuzz_risk['B'] = np.fmin(fuzz_inv['A'], out_risk['B'].mf)

# Figura da agregação
fig, ax0 = plt.subplots()
#
#plt.grid()
uni0 = np.zeros_like(uni)
ax0.fill_between(uni, uni0, fuzz_risk['B'], facecolor='b', alpha=0.7)
ax0.plot(uni, out_risk['B'].mf, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(uni, uni0, fuzz_risk['M'], facecolor='g', alpha=0.7)
ax0.plot(uni, out_risk['M'].mf, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(uni, uni0, fuzz_risk['A'], facecolor='r', alpha=0.7)
ax0.plot(uni, out_risk['A'].mf, 'r', linewidth=0.5, linestyle='--')
plt.ylabel('Pertinência (%)')
plt.xlabel('Risco (%)')
plt.show()

#
# Defuzzificação
 
# Agregando reulstados das inferências (preenchimento)
# Para virar um gráfico só 
aggregated = np.fmax(fuzz_risk['B'],
                     np.fmax(fuzz_risk['M'], fuzz_risk['A']))


# Calculo do centróida (deffuzificação)
risk_defuzz = fuzz.defuzz(uni, aggregated, 'centroid')  # Cálculo do centróide
#risk_value = fuzz.interp_membership(uni, aggregated, risk_defuzz)  # for plot

print('risk (%): ')
print(risk_defuzz)