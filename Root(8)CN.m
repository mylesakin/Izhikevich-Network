N = 64;
n=1000;
Ne = 816; Ni = 208;
p = sqrt(N);
d = sqrt(8);

A = ones(N,N)-eye(N);

x = 1:p;
x = x';
v=x;
y = zeros(p,1);
r = y;
for i = 1:p-1
    x = [x; v];
    r=r+1;
    y = [y; r];
end

for i = 1:N
    for j = 1:N
        e = sqrt((x(i)-x(j))^2+(y(i)-y(j))^2);
        if (e>d)
            A(i,j)=0;
        end
    end
end

%small world no new links
p3 = 0.4; %rewire probability
for i = 1:N
    for j = 1:N
        pn = rand(1,1);
        h=0;
        if (pn<p3) && (A(i,j)==1)
            A(i,j) = 0;
            A(j,i) = 0;
            while (h<1)
                k = randi([1,N],1,1);
                if (A(i,k)==0)
                    A(i,k) = 1;
                    A(k,i) = 1;
                    h = h+1;
                end
            end
        end
    end
end

p1=0.6; %recurrent percent                  
for i = 1:N
    for j = 1:N
        p = rand(1,1);
        if (p > p1) && (A(i,j)==1) && A(j,i)==1
            pr = rand(1,1);
            if (pr>0.5)
                A(i,j)=0;
            end
            if (pr<0.5)
                A(j,i)=0;
            end
        end
    end
end


In = datasample([1:Ne+Ni-1],Ni,'Replace',false);
E = 1:(Ne+Ni);
for i = 1:Ni
    l=length(E);
    for j = 1:l
        if E(1,j)== In(1,i)
            E(E==In(1,i))=[];
            break
        end
    end
end
            
%re = rand(Ne,1); ri = rand(Ni,1);

%a = [0.02*ones(Ne,1); 0.1*ones(Ni,1)]; %column concatenation
%b = [0.2*ones(Ne,1); 0.2*ones(Ni,1)];
c = zeros(Ne+Ni,1);
c(E,1)= -50;%*ones(Ne,1); 
c(In,1)= -45;%*ones(Ni,1);
%d = [8-6*re.^2; 2*ones(Ni,1)];
S = normrnd(1.1,0.01,Ne+Ni,Ne+Ni);
IH = A*.7;
S(:,In)=IH(:,In);
S = A.*S;
uu = [zeros(Ne,1); zeros(Ni,1)];
v = -70*ones(Ne+Ni,1);
u = -20*ones(Ne+Ni,1);
firings = [];
%initital poisson
PT = -log(1-unifrnd(0.3,1,Ne+Ni,1))/0.012;
PT2 = PT-log(1-unifrnd(0.3,1,Ne+Ni,1))/0.012;
Tm = zeros(Ne+Ni,1);
Is = zeros(Ne+Ni,1);
TmSt = zeros(Ne+Ni,n*0.1+1);
TmTm = zeros(Ne+Ni,50);
HH = zeros(Ne+Ni,50);
ctf = ones(Ne+Ni,1);
indE=zeros(Ne+Ni,1);
indI=zeros(Ne+Ni,1);
G=[];
Am = [];
q=1;
SPU = zeros(Ne+Ni,2);
SP = zeros(Ne+Ni,2);
sc = ones(Ne+Ni,1);
H = zeros(Ne+Ni,1);

for t = 1:n
    SPU(:,:)=(t+10);
    PT = round(PT,0);
    PT2 = round(PT2,0);
    Io=zeros(Ne+Ni,1);
    SPi = find(SP>0);
    SPU(SPi) = SP(SPi);
    %synaptic code
    HSE = heaviside(0.1*(t-1)-0.1*SPU(E,:)-5+0.0001);
    HSI = heaviside(0.1*(t-1)-0.1*SPU(In,:)-1+0.0001);
    tt = 0.1*(t-1)-0.1*SPU;
    G = -HSI.*exp(-(tt(In,:)-1)/3);
    Am = HSE.*exp(-(tt(E,:)-5)/3);
    GG = sum(G,2);
    AmA = sum(Am,2);

    %external input
    Is(In) = GG;
    Is(E) = AmA;
    %K = find(PT<=0.1*t);
    Io = 7*heaviside(0.1*(t-1)-PT+0.0001).*exp(-(0.1*(t-1)-PT)/3);
    
    %fired information
    firedE = find(v(E)>=35);
    firedI = find(v(In)>=25);
    
    %Tm(E(firedE))=t*0.1; %fired time for synapses
    %Tm(In(firedI))=t*0.1;
    
    %ti=round((t*0.1+1),0);
    indE = sub2ind(size(TmSt),E(firedE),ctf(E(firedE))');
    indI = sub2ind(size(TmSt),In(firedI),ctf(In(firedI))');
    TmSt(indI) = t*0.1;
    TmSt(indE) = t*0.1; %synase fired time matrix
    SP(E(firedE),sc(E(firedE))) = t;
    SP(In(firedI),sc(In(firedI))) = t;
    sc(E(firedE)) = sc(E(firedE))+1;
    sc(In(firedI)) = sc(In(firedI))+1;
    reset = find(sc==3);
    sc(reset)=1;
    %TmSt(:,t) = Tm;
    %Tm = zeros(Ne+Ni,1);
    ctf(E(firedE)) = ctf(E(firedE))+1;
    ctf(In(firedI)) = ctf(In(firedI))+1;
    indE=zeros(Ne+Ni,1);
    indI=zeros(Ne+Ni,1);
 
    %firings = [firings; t*0.1+0*E(firedE)', E(firedE)'];
    %firings = [firings; t*0.1+0*In(firedI)', In(firedI)'];
    
    %fired = [E(firedE)'; In(firedI)'];
    %Io(In)=0.5*Io(In)
    I=Io+S*Is;
    v(E(firedE)) = c(E(firedE));
    u(E(firedE)) = u(E(firedE))+100;
    v(In(firedI)) = c(In(firedI));
    
    
    %Euler method
    %excitatory
    %v(E) = v(E)+0.05*(0.01*(0.7*(v(E)+60).*(v(E)+40)-u(E))+I(E));
    v(E) = v(E)+0.1*(0.01*(0.7*(v(E)+60).*(v(E)+40)-u(E))+I(E));
    u(E) = u(E)+0.1*(0.03*(-2*(v(E)+60)-u(E)));
    %inhibitory
    %v(In) = v(In)+0.05*(0.05*((v(In)+55).*(v(In)+40)-u(In))+2.5*I(In));
    v(In) = v(In)+0.1*(0.05*((v(In)+55).*(v(In)+40)-u(In))+2.5*I(In));
    u(In) = u(In)+0.1*(0.2*(uu(In)-u(In)));
    
    U = find(v(In)>=-55);
    uu(In(U)) = 0.025*(v(In(U))+55).^3;
    UU = find(v(In)< -55);
    uu(In(UU)) = 0;
    
    %poisson input
    PP = find(PT2==round(0.1*t,0));
    PT(PP) = PT2(PP);
    PT2(PP)= 0.1*t - log(1-unifrnd(0.3,1,length(PP),1))/0.012;
    t
end;
%plot(firings(:,1),firings(:,2),'.');

Z=zeros(N,1);
for i =1:N
    a=find(TmSt(i,:)~=0);
    z(i)=length(a);
end

dl=max(z)
timestamps = zeros(N,dl);
for i =1:N
    timestamps(i,1:z(i))=TmSt(i,find(TmSt(i,:)~=0));
end

    
%dlmwrite('timestamps_ml_1024network_root8.csv', timestamps,'precision','%.6f')
%csvwrite('adjacency_ml_1024network_root8.csv',A)
