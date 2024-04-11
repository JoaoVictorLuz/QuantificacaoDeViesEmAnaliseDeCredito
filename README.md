# Quantificação De Víes Em Análise De Credito 

## Recomendações para rodar o código

Para facilitar a instalação de bibliotecas recomendo utilizar o gerenciador de ambientes [ANACONDA](https://docs.anaconda.com/free/anaconda/install/linux/), aqui está o passo a passo para criar um ambiente e instalar os pacotes necessários 

Para criar o ambiente: 
```
conda create -n aif360 python=3.11
```

Para entrar no ambiente criado(caso ainda não esteja dentro dele)
```
conda activate aif360

```
Depois será necessário apenas executar esses dois comandos para instalar duas bibliotecas: 
```
conda install pip
conda install conda-forge::openpyxl=3.1.2
```


## Avisos Importantes
- Acredito que depois de rodar os experimentos ainda da pra melhorar muito a organização desse código, mas a base está aí, cada experimento está sendo rodado em um arquivo ipynb com o mesmo nome do __dataset__ utilizado  
- Existe uma __channel__  em que é possível instalar o ai360 através do *Conda*, mas não testei essa opção e optei por utilizar o *pip* dentro do cógigo
- Eu utilizo o __vscode__ como editor de código e para rodar os arquivos do tipo ipynb são necessárias extensões dentro do próprio vs code (não sei como isso funciona em outros editores)
- Assim como mencionado anteriormente, o __aif360__ foi instalado dentro do código, curiosamente só essa biblioteca e algumas dependências já são suficiente para rodar os cógigos, supostamente bibliotecas amplamente utilizadas como __numpy__ e __pandas__ já vem inclusas no aif360. Ou eu estou viajando e instalei essas bibliotecas sem querer no ambiente conda





