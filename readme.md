# Multi Class Classification - Alzheimer

## Visualização dos dados:

    No arquivo 'vizualization.py' foi calculado a correlação entre os dados e feito a exibição visual, foi utilizado:

        - Método de Spearman para criar a matriz de correlação
        - Função map do Pandas para transformar os dados em binário e númericos
        - Excluído todos os dados com linhas com contéudo 'NAN'

    Plot da matriz de correlação:

## Data preprocessing:

    Utilizado as funções de OneHotEnconder para onde existe as multiclasses 'Group' e o Enconder para o transformação binária.

    Foi implementado o LabelEnconder apenas para fins demonstrativos da forma de como fazer, caso queira fazer a transformação diretamente no Dataframe com a função Map (como na vizualization.py) o resultado é o mesmo.

## Neural Network:

    