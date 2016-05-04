% Function to compare with Matlab output
m=512; mes=55; n=100; dat=25;
nonzero=[ 65,  74, 102, 111, 191, 206, 221, 225, 228, 235, 250, 251, ...
          254, 293, 302, 305, 312, 340, 388, 397, 400, 406, 421, 434, 477];
f=zeros(m);
f(nonzero)=1;
A = % load Fourier matrix here.



%% EXPORT MATRIX TO CSV IN ORDER TO USE IT WITH MATLAB...
