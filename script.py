import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt

# ASSUMPTIONS

t=1.0     #Total time horizon (years)
s0=100    #  Initial stock price
k=105     # Strike price
r=0.04   #Risk free rate
sigma=0.20   #Average Volatility

M=10000    #Monte Carlo repetions

# Geometric Brownian Motion (GBM)

z=np.random.standard_normal(M)   #it saves in a numpy array   np in the main library/ random is the sub one (focused on prob. and random fucntions) and then we got the specific tool

st=s0*np.exp((r-0.5*(sigma**2))*t+sigma*np.sqrt(t)*z)   # is also saved in array (all the final prices)

payoffs=np.maximum(st-k,0)  # it compares value by value in each array

expected_payoff=np.mean(payoffs)
v0=np.exp(-r*t)*expected_payoff

print(f"Monte Carlo Call Option Value: {v0:.4f}")

#   ------------------------------------


def black_scholes(s0,k,sigma,t,r):
  d1=(np.log(s0/k)+(r+0.5*(sigma**2))*t)/(sigma*np.sqrt(t))
  d2=d1-(sigma*np.sqrt(t))
  v02=s0*norm.cdf(d1)-k*np.exp(-r*t)*norm.cdf(d2)

  return v02

print(f"Black Scholes Call Option Value: {black_scholes(s0,k,sigma,t,r):.4f}")

#Graph Simulation

M_plot=50   #simplified to make the graph more clear
days=252  
dt=t/days
z_paths = np.random.standard_normal((days, M_plot))

daily_return = (r - 0.5 * (sigma**2)) * dt + sigma * np.sqrt(dt) * z_paths  #resultado final (daily_return) será uma nova matriz com exatamente o mesmo tamanho de z_paths (linhas = passos, colunas = n_caminhos), contendo o retorno simulado para cada cenário em cada instante de tempo.
factors = np.exp(daily_return) 

paths = np.zeros((days + 1, M_plot)) #Criar uma matriz vazia só com zeros com 253 linhas (dia 0 até ao dia 252) e 50 colunas
paths[0] = s0 #No dia 0 (linha 0), todas as 50 trajetórias começam em s0
paths[1:] = s0 * np.cumprod(factors, axis=0)  # Do dia 1 para a frente, acumula as variações diárias axis=0: Refere-se às linhas (direção vertical).


plt.figure(figsize=(10, 6))  # criar uma nova janela de gráfico vazia e definir o seu tamanho físico (opcional)  figsize não é uma função independente, mas sim uma propriedade (ou argumento) pertencente à função plt.figure()  -- este cria a janela do graficxo.
plt.plot(paths, linewidth=0.8, alpha=0.6)    #plt.plot(dados_do_eixo_x, dados_do_eixo_y, argumento1=valor1, argumento2=valor2, ...)  aqui os dados do x sirgem automaciamente como limhas a matriz paths
plt.axhline(s0, color="black", linestyle=":", linewidth=1.2, label=f"Preço Inicial S0 ({s0}€)")     # axhline +e deseha linhas horizontais
plt.axhline(k, color="red", linestyle="--", linewidth=1.5, label=f"Strike K ({k}€)")
plt.title("Monte Carlo Simulation Graph", fontsize=12)
plt.xlabel("Dias Úteis de Mercado", fontsize=10)
plt.ylabel("Preço da Ação (€)", fontsize=10)
plt.show()  # Mostra a janela com o gráfico
          
