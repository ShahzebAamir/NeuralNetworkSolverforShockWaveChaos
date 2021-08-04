%%
clear;
clc;
In = readtable('In.csv');
Out = readtable('Out.csv');
T = readtable('Time.csv');
In = table2array(In);
Out = table2array(Out);
T = table2array(T);
In = In(:,2);
Out = Out(:,2);
T = T(:,2);
U = [In; Out(end)];
U = U(8000:12000);
T = T(8000:12000);
N = length(U);
Nu = round(0.7*N);
Utrain = con2seq(U(1:Nu)');
Uval = con2seq(U(Nu+1:end)');

plot(T(1:Nu),U(1:Nu),'m');
hold on;
plot(T(Nu+1:N),U(Nu+1:end),'b');
e=5;
%%
while e>2
inputDelays = 1:2:5; % input delay vector??
%for i=1:5
%hiddenSizes(i) = randi([5 15]); % network structure (number of neurons)
%end
rng(1);
hiddenSizes = [15 15 15 15 15 15];
net = narnet(inputDelays, hiddenSizes, 'open');
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.9;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.0; 
for j=1:5
net.layers{j}.transferFcn = 'logsig';
end
[Xs,Xi,Ai,Ts] = preparets(net,{},{},Utrain);
%rng('default');
net = train(net,Xs,Ts,Xi,Ai);
%plotting
net = closeloop(net);
Uval_ini = Utrain(end-max(inputDelays)+1:end);
[Xs,Xi,Ai] = preparets(net,{},{},[Uval_ini Uval]);
predict = net(Xs,Xi,Ai);
% validation data
Yv = cell2mat(Uval);
% prediction
Yp = cell2mat(predict);
e = norm((Yv - Yp),2);
display(e);
plot(T(Nu+1:N),Yv,'r');
hold on;
plot(T(Nu+1:N),Yp,'b');
hold off;
end
%%
plot(T(Nu+1:end),Yv,'r');
hold on;
plot(T(Nu+1:end),Yp,'b');
%axis([0 3500 0.5 1.5])
%figure;
%plot(T(Nu+1:N),e,'r-')