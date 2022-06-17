% This file does the work. It will genreate all the data files given the bounds of the particle, the number of examples to generate, and the number of layers in the particle. It will save the data into the data/ folder. 
%
addpath 'spherical_T_matrix';
addpath 'spherical_T_matrix/bessel';

low_bound = 30;
up_bound = 70;
num_iteration = 1000;

num_layers = 5;

values = zeros(num_iteration,num_layers);
myspects = zeros(201,num_iteration);

tic
for n = 1:num_iteration
  r = zeros(1,num_layers);
  for i = 1:num_layers
    r1 = round(rand*(up_bound-low_bound)+low_bound,1);
    r(i) = r1;
  end
  spect = scatter_sim_0_gen_single_spect(r);
  myspects(:,n) = spect(1:2:401,1);
  values(n,:) = r;
  if rem(n, 100) == 0
    disp('On: ')
    disp(n)
    disp(num_iteration)
  end
end
toc

% csvwrite(strcat('data/',num2str(num_layers),'_layer_tio2.csv'),myspects);
% csvwrite(strcat('data/',num2str(num_layers),'_layer_tio2_val.csv'),values);