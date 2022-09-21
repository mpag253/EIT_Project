addpath('../distmesh')

clear, clc, close all

% % Example: (Ellipse)
% fd=@(p) (p(:,1).^2)/((a)^2)+(p(:,2).^2)/((b)^2)-1;
% % [p,t]=distmesh2doriginal(fd,@huniform,0.1,[-2,-1;2,1],[]);
% [p,t]=distmesh2doriginal(fd,@huniform,10,[-187,-131;187,131],[]);

a = 187; % for LOO-A
b = 131; % for LOO-A

mesh_size_t = 2;  % torso
mesh_size_i = 8;  % interior
refine_grad = 0.3;  % slow=0.1, fast=0.3

% circumference of ellipse
h = (a-b)^2/(a+b)^2;
p = 0;
binom = @(n,r) gamma(n+1)/gamma(r+1)/gamma(n-r+1);
for i = 0:10, p = p + pi*(a+b)*(binom(.5,i)^2)*(h^i); end

% number of boundary points
n_pts = round(p/mesh_size_t)

% ellipse points
r_e = @(theta) a*b./((b*cos(theta)).^2 + (a*sin(theta)).^2).^.5;
t_e = linspace(0,2*pi,1000);
x_e = r_e(t_e).*cos(t_e);
y_e = r_e(t_e).*sin(t_e);

pv_t = interparc(n_pts,x_e,y_e,'spline');
% plot(pv_t(:,1),pv_t(:,2),'o'), hold on
% plot(x_e,y_e)
% axis equal
pv_all = pv_t;

bbox = 1.01*[min(pv_t(:,1)) min(pv_t(:,2)); max(pv_t(:,1)) max(pv_t(:,2))];
fd = @(p) dpoly(p, pv_t); %@dpoly;
fh = @(p) min(mesh_size_t-refine_grad*dpoly(p, pv_t), mesh_size_i);
h0 = min(mesh_size_t,mesh_size_i);
fix = pv_all;
[nodes,tris]=distmesh2d(fd,fh,h0,bbox,fix);

tgeom = triangulation(tris,nodes);
% tgeom = delaunayTriangulation(Nodes);
% Tris = tgeom.ConnectivityList;

% boundary_indices = boundary(nodes(:,1), nodes(:,2), shrink_factor);
% boundary_operator(boundary_indices) = true;
% boundary_operator = boundary_operator';

mesh_specifics = strcat(sprintf('%04d',10*mesh_size_i),'_',sprintf('%04d',10*mesh_size_t));
% save_path = strrep(path, 'mesh_seeds/', strcat('trimesh/trimesh_',mesh_specifics,'/'));
save_path = strcat('pca_LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-A/ellipse/trimesh/trimesh_',mesh_specifics,'/');
if ~exist(save_path, 'dir'), mkdir(save_path), end
save(strcat(save_path, 'trimesh.mat'),'tgeom')
save_file_nods = strcat('trimesh_nodes.csv');
save_file_tris = strcat('trimesh_nodes.csv');
writematrix(tgeom.Points, strcat(save_path,save_file_nods))
writematrix(tgeom.ConnectivityList, strcat(save_path,save_file_tris))

disp("Done.")