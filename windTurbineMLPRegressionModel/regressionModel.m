% Emanuel Valério Pereira - Matricula : 471055

% ************ MODELO DE REGRESSÃO USANDO REDE NEURAL MLP ***********%
base = load('aerogerador.dat');
 
x = base(:,1)'; % Valores de entrada - Velocidade do Vento
y = base(:,2)'; % Valores de Saída   - Potencia Gerada

% Rede MLP

layerHidden01 = 5;  % Quantidade de Neuronios na camada oculta
layerHidden02 = 10; % Quantidade de Neuronios na camada oculta
layerOut = 1;
%newNetwork = newff(x,y,[layerHidden01,layerOut],{'tansig','purelin'}); % Exemplo com apenas 1 camada oculta;

newNetwork = newff(x,y,[layerHidden01,layerHidden02,layerOut],{'tansig','tansig','purelin'}); 

 % Sintaxe : newff(x,y,Si,TF,BTF,BLF,PF,IPF,OPF,DDF);
 % x é a base de dados de entrada
 % y é a potencia gerada
 % Si são a quantidades de neurônios por camadas, S1,S2...Sn
 % TF Funções de ativação das camadas, tansig nas camadas ocultas
 % (Multidimensionais) e pirelin uma função linear na camada de saida.
 % BTF - Função de backpropagation que por padrão é 'trainml'.
 % Outros parametros ficaram com seu valores padrões.

 % Optei por usar duas camadas ocultas, apenas para representar a ideia de
 % multipla camada, mas obtive bons resultados com apenas uma camada
 % oculta.

     net.trainParam.lr = 0.005;  % Taxa de aprendizado
     net.trainParam.epochs = 20; % Número de épocas
     net.trainParam.goal = 0.01; % Erro admitido
    
    % Treinamento da Rede
    [newNetwork,tr] = train(newNetwork,x,y);
    % Sintaxe : train(rede_neural,entrada,saida_desejada);

    %Teste da Rede
    y_teste = sim(newNetwork,x); % sim simula uma rede neural
    % Sintaxe: sim(rede_treinada,entradas_desejadas);
    R2 = ( 1 - (sum((y-y_teste).^2) / sum((y-mean(y)).^2))); % coeficiente de determinacao R2 do modelo;

    figure,plot(x,y,'k*');
    hold on
    plot(x,y_teste,'r-');
    hold off
    
    % Avaliando a qualidade do modelo pela métrica R2
    fprintf("R2  : %f\n",R2);