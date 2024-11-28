
# Nome do Projeto

**Descrição**: Este projeto é um sistema interativo para análise de dados e predição de risco de AVE (Acidente Vascular Encefálico) utilizando aprendizado de máquina. Ele combina uma interface intuitiva baseada em Flask com modelos de classificação como Random Forest, SVM e Gradient Boosting para realizar análises preditivas. O objetivo é fornecer insights precisos e acessíveis sobre a probabilidade de AVC com base em características comportamentais e clínicas.



---

## 📋 Sumário

1. [Visão Geral](#visão-geral)
2. [Funcionalidades](#funcionalidades)
3. [Tecnologias Utilizadas](#tecnologias-utilizadas)
4. [Requisitos](#requisitos)
5. [Configuração e Instalação](#configuração-e-instalação)
6. [Como Usar](#como-usar)
7. [Estrutura do Projeto](#estrutura-do-projeto)
8. [Contribuição](#contribuição)
9. [Licença](#licença)
10. [Contato](#contato)

---

## 📖 Visão Geral

O projeto foi desenvolvido para analisar e prever o risco de Acidente Vascular Cerebral (AVC) em pacientes. Ele busca fornecer uma ferramenta acessível e interativa para profissionais da saúde e pesquisadores realizarem análises preditivas baseadas em características clínicas e comportamentais. O sistema atende a cenários em que é necessário identificar fatores de risco, apoiar a tomada de decisões médicas e priorizar ações preventivas para pacientes com alta probabilidade de desenvolver AVC.

---

## 🚀 Funcionalidades

- **Upload e análise de dados clínicos**: Permite o upload de arquivos CSV contendo informações sobre pacientes para análise de fatores de risco relacionados ao AVC. O sistema processa e exibe gráficos interativos para auxiliar na interpretação dos dados.
- **Treinamento de modelos de machine learning**: Oferece suporte para treinar modelos preditivos, como Random Forest, SVM e Gradient Boosting, com opções de personalização de parâmetros, permitindo a escolha do melhor modelo para prever a ocorrência de AVC.
- **Predição de risco de AVC**: Permite que os usuários preencham características de um paciente em um formulário interativo. Com base nos modelos treinados, o sistema realiza predições, indicando a probabilidade do paciente apresentar risco de AVC.
- **Geração de gráficos preditivos** Gera gráficos como a matriz de confusão e a curva ROC para avaliar a eficácia dos modelos preditivos, fornecendo insights claros sobre a precisão e confiabilidade do sistema.

---

## 🛠 Tecnologias Utilizadas

- **Linguagem**: Python
- **Bibliotecas/Frameworks**: 
    - **Pandas**: Para manipulação e análise de dados tabulares.
    - **Seaborn** e matplotlib: Para visualização de dados incluindo gráficos e métricas do modelo.
    - **Flask**: Para construção da interface web, permitindo interatividade e facilidade de uso.
    - **Scikit**-learn: Para implementação de modelos de machine learning e geração de métricas de avaliação.
    - **Joblib**: Para salvar e carregar modelos treinados.
    - **Logging**: Utilizada para capturar, registrar e exibir informações relevantes sobre a execução do programa, como mensagens de erro, avisos, informações e depuração. 
    - **uuid**: A biblioteca uuid é usada para gerar identificadores únicos universais (UUIDs), garantindo que nomes de arquivos ou identificadores gerados sejam exclusivos.
- **Banco de Dados**: Não aplicável
- **Ferramentas**: 
    - **Poetry**: Utilizado como gerenciador de dependências e ferramenta de automação para o ambiente Python. Ele permite a instalação, atualização e remoção de bibliotecas de forma eficiente, além de ajudar na criação e publicação de pacotes Python.

---

## 📋 Requisitos

- **Python 3.12** ou superior
- Dependências adicionais (instaladas via `pyproject.toml`)

---

## 🖥️ Configuração e Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/JOAOGOMES22/Stroke-Prediction.git
   ```
2. Navegue até o diretório do projeto:
   ```bash
   cd Stroke-Prediction
   ```
3. Instale as dependências:
   ```bash
   poetry install
   ```
4. Ative o ambiente virtual gerado pelo Poetry:
   ```bash
   poetry shell
   ```
---

## 📌 Como Usar

1. Execute o script principal:
   ```bash
   python main.py
   ```
2. Acesse a interface web.

3. Para rodar os testes, utilize (TO-DO):
   ```bash
   pytest tests/
   ```

---

## 🗂 Estrutura do Projeto

```plaintext
├── app/                 # Diretório da aplicação
│   ├── static/          # Arquivos estáticos (CSS, imagens, etc.)
│   │   ├── styles.css   # Estilos personalizados
│   │   ├── *.png        # Imagens geradas para gráficos e resultados
│   ├── templates/       # Templates HTML para renderização
│   │   ├── index.html   # Página principal
│   ├── uploads/         # Arquivos carregados pelo usuário
│       ├── *.csv        # Dados de entrada para análise e treinamento
│
├── models/              # Modelos treinados
│   ├── model.joblib     # Arquivo do modelo salvo
│
├── src/                 # Código-fonte principal
│   ├── config.py        # Configurações do projeto (colunas selecionadas, etc.)
│   ├── data_analysis.py   # Funções para geração de gráficos e análise
│   ├── data_processing.py # Pipeline de pré-processamento de dados
│   ├── model_training.py  # Algoritmos de treinamento e predição
│   ├── utils.py         # Funções auxiliares
│   └── __init__.py      # Inicialização do módulo
│
├── tests/               # Testes automatizados
│   ├── test_*           # Arquivos de teste
│
├── .gitignore           # Arquivos ignorados pelo Git
├── main.py              # Script principal para execução do projeto
├── poetry.lock          # Arquivo de bloqueio de dependências do Poetry
├── pyproject.toml       # Configurações do Poetry e dependências do projeto
└── README.md            # Documentação do projeto
```

---

## 🤝 Contribuição

Sinta-se à vontade para contribuir com este projeto. Para começar:

1. Faça um fork do repositório.

2. Crie um branch para sua feature:
   ```bash
   git checkout -b minha-feature
   ```
3. Faça suas alterações e commit:
   ```bash
   git commit -m "Descrição da feature"
   ```
4. Envie um pull request.

---

## 📝 Licença

Este projeto é licenciado sob a [Nome da Licença] - veja o arquivo [LICENSE](./LICENSE) para mais detalhes.

---

### 🌟 Seja bem-vindo(a) e aproveite!
