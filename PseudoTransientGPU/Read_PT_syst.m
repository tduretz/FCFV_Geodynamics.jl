clear

load('PT_syst')

n  = [1, 2, 4, 8, 10, 6, 1.5, 3, 0.5, 0.75, 1.5, 2.5, 1.25];
dt = zeros(13,1);  dt(5) = 0.256;   dt(6) = 0.238; dt(7) = 0.1575; dt(8) = 0.21;
t  = zeros(13,1);  t(5)  = 0.0115;  t(6) = 0.02;   t(7)  = 0.082;  t(8)  = 0.04;

t(9) = 0.19; 
dt(9) = 0.08;
t(10) = 0.16; 
dt(10) = 0.11;
t(11) = 0.08; 
dt(11) = 0.155;
t(12) = 0.05; 
dt(12) = 0.195;
t(13) = 0.095;
dt(13) = 0.145;


iter(success==0) = 2e4;

figure(1), clf
subplot(221), hold on
it = reshape(iter(1,:,:), 19, 19);
imagesc(tet, dtau, it), colormap;
[i,j,v] = find(it==min(it(:)), 1,'last');
plot(tet(j), dtau(i), 'xw')
dt(1) = dtau(j);
t(1) = tet(i);
xlabel('dt'), ylabel('t'), title(min(it(:)))

subplot(222), hold on
it = reshape(iter(2,:,:), 19, 19);
imagesc(tet, dtau, it), colormap;
[i,j,v] = find(it==min(it(:)));
plot(tet(j), dtau(i), 'xw')
dt(2) = dtau(j);
t(2) = tet(i);
xlabel('dt'), ylabel('t'), title(min(it(:)))

subplot(223), hold on
it = reshape(iter(3,:,:), 19, 19);
imagesc(tet, dtau, it), colormap;
[i,j,v] = find(it==min(it(:)));
plot(tet(j), dtau(i), 'xw')
dt(3) = dtau(j);
t(3) = tet(i);
xlabel('dt'), ylabel('t'), title(min(it(:)))

subplot(224), hold on
it = reshape(iter(4,:,:), 19, 19);
imagesc(tet, dtau, it), colormap;
[i,j,v] = find(it==min(it(:)));
plot(tet(j), dtau(i), 'xw')
dt(4) = dtau(j);
t(4) = tet(i);
xlabel('dt'), ylabel('t'), title(min(it(:)))

% Correction
dt(1) = 0.125;
t(1) = 0.12;

dt(2) = 0.18;
t(2) = 0.06;

dt(3) = 0.22;
% t(3) = 0.06;

%%%%%%%%%%%%%%%%
n1  = 0.02:0.01:20;
L   = 1;
dA  = L./(20*n)/1.5;

dA1 = L./(20*n1)/1.5; % n -> 1.5*dA/L*20
dt1 = 3.5*dA.^(1);

Om1 = 1/2*dA1.^2;

D = 1;
dt1  = dA1.^2/(2*D)/2 .* 3./Om1.^(1);


% quads
nq = [1, 2, 4, 8];
tq = [0.19, 0.11, 0.057, 0.029];

% quads

figure(2), clf
subplot(131), hold on
p = polyfit(log10(n), log10(dt), 1);
% plot( log10(n1), p(2) + log10(n1).*p(1))
% plot(log10(n), log10(dt), 'o')
% plot( (dA1), exp(p(2)) .* (dA1).^p(1))
plot( 1./(1.5*dA/L*20), (dt), 'o')
plot(n1, 0.08*(L./(20*n1)/1.5).^(-0.2))
% % axis([0 .1 0 0.5])
title('dtau'), xlabel('n')
subplot(132), hold on
p = polyfit(log10(dA), log10(dt), 1);
% plot( log10(dA1), p(2) + log10(dA1).*p(1))
% plot(log10(dA), log10(dt), 'o')
% plot( (dA1), exp(p(2)) .* (dA1).^p(1))
plot((dA), (dt), 'o')
% plot(dA1, -4.5*dA1 + 0.265)
plot(dA1, dt1)
% axis([0 .1 0 0.5])
title('dtau'), xlabel('dA')
subplot(133), hold on
p = polyfit(log10(n), log10(t), 1);
% plot( log10(n), p(2) + log10(n).*p(1))
% plot(log10(n), log10(t), 'o')
plot(n, t, 'o')
plot(nq, tq, 'x')
ncx = 20*n1;
plot(n1, 7.10*2 * L./pi./ncx, '-k')
plot(n1, 7.4074 * L./pi./ncx)
% plot( (n1), exp(p(2)*2.3) .* (n1).^p(1))
axis([0 20 .01 .3])
title('theta'), xlabel('ncx')