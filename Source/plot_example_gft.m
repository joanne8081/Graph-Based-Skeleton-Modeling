% Plot the graph Laplacian eigenvectors of an example graph

% Add GSPBox
addpath('..\Util\gspbox');
gsp_start();

% Create example graphs
% G = gsp_graph(W);
% param.distribute = 1;
% G = gsp_sensor(15, param);
% figure;
% gsp_plot_graph(G);

%% Example graph 2
W = [0 1 0 0 0 1 0 0 ;
     1 0 1 0 0 1 0 0 ;
     0 1 0 1 0 1 1 0 ;
     0 0 1 0 1 0 1 1 ;
     0 0 0 1 0 0 0 1 ;
     1 1 1 0 0 0 1 0 ;
     0 0 1 1 0 1 0 1 ;
     0 0 0 1 1 0 1 0 ];
Coord = [0.2 0.1;
         0.3  0.2;
         0.5  0.2;
         0.7  0.2;
         0.9  0.15;
         0.4  0;
         0.6  0;
         0.8  0];
limit = [0.1 1 -0.2 0.4];
figure;
G = gsp_graph(W, Coord, limit);
axis off
gsp_plot_graph(G);

%% Compute GFT basis
% G = gsp_create_laplacian(G, 'normalized');
G = gsp_compute_fourier_basis(G);
plot_evec_ind = [1 2 4 7];
G.plotting.vertex_size = 1000;
G.plotting.vertex_color = 'k';
G.plotting.edge_color = [0,0,0];
G.plotting.edge_width = 1;

param.colorbar = 0;
param.bar = 1;
param.bar_width = 2;

figure;
for ind=1:4
    subplot(1,4,ind);
    gsp_plot_signal(G, G.U(:,plot_evec_ind(ind)), param);
    title(['\lambda = ' sprintf('%.02f',G.e(plot_evec_ind(ind)))], 'FontSize', 18);
    axis equal; %axis tight;
    axis([0.1 1 -0.2 0.4 -0.7 0.7]);
    view([0 75]);
end


