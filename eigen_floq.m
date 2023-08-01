function eigen_floq(fname, omega, mu, M, EigfSaveFlag)
%function explore_eigfloquet(fname, Np, Omega, mu, M, EigfSaveFlag)
% computes _M eigenvalues nearest to i*_Omega by shift-invert method
% for linearization around a Stokes wave loaded from file _fname with 
% Np subharmonics with Floquet parameter _mu. 
% EigfSaveFlag can be 1 -- save eigenfunctions,
% 	 	      0 -- discard eigenfunctions (to save space on hdd)	        

maxNumCompThreads(1);
nthreads = maxNumCompThreads

% take Np periods (must be a power of 2) of Stokes wave:
%Np = 512;
% imaginary shift
[y, N, c] = load_stokes(fname);
yn = zeros(N,1);
tmp = fft(y)/N;
yn(1:N/2) = 2*tmp(1:N/2);
yn(1) = 0.5*yn(1);
y = real(ifft(yn)*N);
yk = fft(y)/N;

k = fftshift(-N/2:N/2-1)';
u = pi*(2*(0:N-1)/N -1)';
x = u - ifft(1i*sign(k).*fft(y), 'symmetric');
Dx = 1 + ifft(abs(k).*fft(y), 'symmetric');
Dy = ifft(1i.*k.*fft(y), 'symmetric');
nsqr = (Dy'*Dy);

% check first and second form of Stokes Equation:
% Res1 = c.^2*ifft(abs(k).*fft(y)) - y - (y.*ifft(abs(k).*fft(y)) + 0.5*ifft(abs(k).*fft(y.^2)));
% Res2 = y - 0.5*c^2 + 0.5*c^2./(Dx.^2 + Dy.^2);

% dy or f -- in1; dphi or g -- in2
in1 = 0.12*cos(u) + 0.25*sin(u);  
in2 =-c.*in1 + 0.18*sin(3*u) + 0.4*cos(4*u);

in = zeros(2*N, 1);
in(1:N) = in1;
in(N+1:2*N) = in2; 

% number of eigenvalues to compute
%M = 512;
HL = 0.5*(max(y) - min(y))/pi;
[V, D, ~] = eigensabs2(in, y, c, M, omega, mu);
eignm = sprintf('library_mu/HL_%0.10f.SI.subh', HL);
fhlog = fopen(eignm, 'r');
if (fhlog == -1) 
 fhlog = fopen(eignm,'w');
 fprintf(fhlog, '# 1. number 2.-3. eigenvalue 4. mu\n\n');
 for j = 1:M
    fprintf(fhlog, '%5d\t%22.15e\t%22.15e\t%22.15e\n', j, real(D(j,j)), imag(D(j,j)), mu);
 end
 fclose(fhlog);
else 
 fclose(fhlog);
 fhlog = fopen(eignm, 'a');
 fprintf(fhlog, '\n\n');
 for j = 1:M
    fprintf(fhlog, '%5d\t%22.15e\t%22.15e\t%22.15e\n', j, real(D(j,j)), imag(D(j,j)), mu);
 end
 fclose(fhlog);
end

% eigenfunction list:
% - normalize so that max abs y_k = 1
if (EigfSaveFlag)
  for j = 1:M
    eignm = sprintf('library_mu/eigf/HL_%0.10f_mu_%0.10f_n%03d.subhf', HL, mu, j);
    fh = fopen(eignm, 'w');
    fprintf(fh, '%%# 1. u 2. x 3.-4. dy 5.-6. dp (not dpsi) 7. y 8.-9. dx = -Hdy\n');
    fprintf(fh, '%%# Eig # %04d, Re lambda = %.12e\tIm lambda = %.12e\n\n', j, real(D(j,j)), imag(D(j,j)) );
    % normalize to real peak
    [~, ia] = max(abs(V(N+1:2*N,j)));
    V(:,j) = V(:,j)/V(N+ia, j);
    dx = -ifft(1i*sign(k+mu).*fft(V(N+1:2*N,j)));
    for l = 1:N
        % normalize to unit Fourier coefficient
        %dYk = fft(V(N+1:2*N,j))/N;
        %[a, ia] = max(abs(dYk));
        %V(:,j) = V(:,j)*conj(dYk(ia))/a^2;
        
        fprintf(fh, '%22.15e\t%22.15e\t%22.15e\t%22.15e\t%22.15e\t%22.15e\t%22.15e\t%22.15e\t%22.15e\n', ...
            u(l), x(l), real(V(N+l, j)), imag(V(N+l, j)), real(V(l, j)), imag(V(l, j)), y(l), real(dx(l)), imag(dx(l)));
    end
    fclose(fh);
  end
end

end

function [V, D, flag] = eigensabs2(in, y, c, d, omega, mu)
  N = length(y);
  k = fftshift(-N/2:N/2-1)';
  u = pi*(2*(0:N-1)/N -1)';
  yk = fft(y);
  x = u - ifft(1i*sign(k).*yk, 'symmetric');
  Ky = ifft(abs(k).*yk, 'symmetric');
  Dy = ifft(1i.*k.*yk, 'symmetric');
  v0 = in;
  c1 = sum(v0(1:N).*(1 + 2.*Ky))/sum((1 + 2.*Ky).^2);
  v0(1:N) = v0(1:N) - c1*(1 + 2.*Ky); 
  
   
  opts.disp = 1; % 0 reduced output
  opts.v0 = v0;
  %[V, D, flag] = eigs(@afun, 2*N, d, 1i*omega, 'Display', 1,'StartVector', v0);
  [V, D, flag] = eigs(@afun, 2*N, d, 1i*omega, opts);

  function outA = operator_J(inA)
      outA = zeros(2*N,1); t = outA;
      %t = inA(N+1:2*N);
      t = inA(N+1:2*N);
      outA(1:N) = t;
      outA(N+1:2*N) = inA(1:N) - 2.*c*ifft(1i*sign(k+mu).*fft(t));
  end
  function outA = afun(inA)
    tmp =  operator_J(inA);
    outA = operator_shift_invert(tmp, y, c, omega, mu);
    %mean_dp = fft(outA(1:N))/N; is nonzero
    %t = fft(outA(1:N)); t(1) = 0;
    %outA(1:N) = ifft(t);
    %mean_dy = fft(outA(N+1:2*N))/N;
    
    %tmp2 = fft(outA(1:N)); 
    %tmp2(1) = 0;
    %tmp2 = ifft(tmp2);
    %outA(1:N) = tmp2;
     
    %tmp2 = fft(outA(N+1:2*N)); 
    %tmp2(1) = 0;
    %tmp2 = ifft(tmp2);
    %outA(N+1:2*N) = tmp2;
  end
end

function out = operator_S2(in, y, c, omega, mu)
  N = length(y);
  k = fftshift(-N/2:N/2-1)';
  Dx = 1 + ifft(abs(k).*fft(y), 'symmetric');
  
  function outA = operator_S1(inA)
    outA = c^2*ifft(abs(k+mu).*fft(inA)) -  ...
           Dx.*inA - y.*ifft(abs(k+mu).*fft(inA)) - ...
           ifft(abs(k+mu).*fft(y.*inA));
  end
  out = operator_S1(in) - 2.*c*omega*ifft(sign(k+mu).*fft(in));
end
function out = operator_shift_invert(in, y, c, omega, mu)
  N = length(y);
  k = fftshift(-N/2:N/2-1)';
  yk = fft(y);
  Ky = ifft(abs(k).*yk, 'symmetric');
  Dx = 1 + Ky;
  Dy = ifft(1i.*k.*yk, 'symmetric');  
  f = in(1:N);
  g = in(N+1:2*N);
  out = zeros(2*N,1);
  function outA = operator_Q(inA)  % missing zero mode here
      % prepare input to have mean zero:
      if (mu == 0)
        dy0 = sum((1. + 2.*Ky).*inA)/N;
        inA = inA - dy0;
      end
      tmp = operator_W21(inA); 
      tmpk = fft(tmp); %tmp0 = tmpk(1);
      tmpk = tmpk./abs(k+mu);
      phi01 = 0;
      if (mu == 0) 
        tmpk(1) = 0;
        phi01 = sum(abs(k).*tmpk.*yk)/N/N;
      end
      tmp = ifft(tmpk)-phi01;
      outA = -operator_W12(tmp);
  end
  function outA = operator_W12(inA)
      outA = -Dx.*inA + ifft(1i*sign(k+mu).*fft(Dy.*inA));
  end
  function outA = operator_W21(inA)
      outA = Dx.*inA + Dy.*ifft(1i*sign(k+mu).*fft(inA));
  end

  Qf = operator_Q(f);
  t = g + 1i*omega*Qf;
  t2 = operator_A(t, y, c, omega, mu);
  t1 = operator_Q(t2);
  out(1:N) = Qf + 1i*omega*t1;
  out(N+1:2*N) = t2;
    
end
function out = operator_A(in, y, c, omega, mu)
  N = length(y);
  k = fftshift(-N/2:N/2-1)';
  yk = fft(y);
  Ky = ifft(abs(k).*yk, 'symmetric');
  Hk  = 1./(c^2*abs(k+mu) - 2.*c*omega*sign(k+mu) - 1);
  Hmk = 1./sqrt(c^2*abs(k+mu) + 2.*c*omega + 1);
  Dx = 1 + ifft(abs(k).*fft(y), 'symmetric');
  Dy = ifft(1i.*k.*fft(y), 'symmetric');
    function outA = operator_Q(inA) % zero mode also needs fixing here
      if (mu == 0)
        dy0 = sum((1. + 2.*Ky).*inA)/N;
        inA = inA - dy0;
      end
      tmp = operator_W21(inA);
      tmpk = fft(tmp); %tmp0 = tmpk(1);
      tmpk = tmpk./abs(k+mu);
      phi01 = 0;
      if (mu == 0)
        tmpk(1) = 0;
        phi01 = sum(abs(k).*tmpk.*yk)/N/N;
      end
      tmp = ifft(tmpk)-phi01;
      outA = -operator_W12(tmp);
  end
  function outA = operator_W12(inA)
      outA = -Dx.*inA + ifft(1i*sign(k+mu).*fft(Dy.*inA));
  end
  function outA = operator_W21(inA)
      outA = Dx.*inA + Dy.*ifft(1i*sign(k+mu).*fft(inA));
  end
  function outA = afun_fourierH(inAk)
      inA = ifft(Hmk.*inAk)*N;
      tmp1 = operator_S2(inA, y, c, omega, mu);
      tmp2 = operator_Q(inA);
      outA = Hmk.*fft(tmp1 + omega^2*tmp2)/N;
  end

% set tol and niter
  tol = 2e-11; % 5e-13 is good for N = 16384, H/L = 0.13946...
  	          % 6e-11 is good for N = 65536, H/L = 0.140600...
  maxit = N;

  ink = fft(in)/N;
  outH = minres(@afun_fourierH, Hmk.*ink, tol, maxit);
  out = ifft(Hmk.*outH)*N;
  
end
function [y, N, c] = load_stokes(stokes_file)
  fhin = fopen(stokes_file, 'r');
  if (fhin == -1)
    fprintf('Cannot open Stokes wave file\n');
    %exit(0); 
  else 
    line = fgets(fhin);
    line = fgets(fhin);
    % Octave
    %[v1, v2] = sscanf(line, "# N = %s\tL = %s\n","C");
    %N = str2num(v1) 
    %l  = str2double(v2);
    % Matlab 
    v = sscanf(line, '%%# N = %d\tL = %f\n');    
    N = v(1);
    l = v(2);
    % Advanpix
    % N = mp(v(1));
    % l = mp(v(2));
    
    line = fgets(fhin);
    % Octave
    %[v1, v2, v3, v4] = sscanf(line, "# H/L = %s\tc = %s\ty0 = %s\tOmega = %s\n","C");
    %c = str2num(v2);
    % Matlab
    v = sscanf(line, '%%# H/L = %f\tc = %f\ty0 = %f\tOmega = %f\n');
    c = v(2);
    % Advanpix
    % c = mp(v(29:56));
    
    fprintf('Stokes speed c = %12.8e\n', c); 
    fclose(fhin); 
  end
  % must be replaced with a better read compatible with Advanpix:
  raw = load(stokes_file);
  % Advanpix
  % raw = mp_load_data(stokes_file);
  
  y = raw(:,3);
  %z_u = ifft(1.i*k.*fft(z_tilde)) + 1;
  %R = 1./z_u; 
end
