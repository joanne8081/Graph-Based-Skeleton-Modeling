% Plot typical skeletal graph GFT basis

Ns = 15;
Es = [ 2 2 2 2 4 5 7 8 3 10 11 3 13 14;
          1 3 4 7 5 6 8 9 10 11 12 13 14 15];
As = zeros(Ns , Ns);
for n=1:size(Es,2)
   As(Es(1,n), Es(2,n)) = 1;
   As(Es(2,n), Es(1,n)) = 1;
end
Ls = diag(As*ones(size(As,2),1))-As;
[V,D]=eig(Ls);

% initialize the skeleton
skeleton_loc = zeros(15,2);
skeleton_loc(1,:)=[0 2.5];
skeleton_loc(2,:)=[0 1.5]; 
skeleton_loc(3,:)=[0 0.5];
skeleton_loc(4,:)=[0.4 1.5];
skeleton_loc(5,:)=[0.8 1.1];
skeleton_loc(6,:)=[1.2 1.2];
skeleton_loc(7,:)=[-0.4 1.5];
skeleton_loc(8,:)=[-0.8 1.1];
skeleton_loc(9,:)=[-1.2 1.2];
skeleton_loc(10,:)=[0.3 0];
skeleton_loc(11,:)=[0.4 -0.85];
skeleton_loc(12,:)=[0.4 -1.75];
skeleton_loc(13,:)=[-0.3 0];
skeleton_loc(14,:)=[-0.4 -0.85];
skeleton_loc(15,:)=[-0.4 -1.75];

for bIdx=1:Ns
    figure;
    % first plot the fixed limbs
    for m=1:size(Es,2)
        line( [skeleton_loc(Es(1,m),1), skeleton_loc(Es(2,m),1)],  [skeleton_loc(Es(1,m),2), skeleton_loc(Es(2,m),2)],...
        'LineStyle', '-', 'Color','k','LineWidth',2);
    end
    % plot all the joints
    for m=1:15
        if V(m,bIdx)>0
            line(skeleton_loc(m,1),skeleton_loc(m,2),'marker','s', 'markersize',13,'MarkerFaceColor','b','color', 'b');
        else
            line(skeleton_loc(m,1),skeleton_loc(m,2),'marker','.', 'markersize',38,'MarkerFaceColor','r','color', 'r');
        end
    end
    title(['\lambda = ' sprintf('%.02f', abs(D(bIdx,bIdx)))], 'FontSize', 40);
    axis off
end

%% Plot SK-T-G GFT basis
Ns = 15;
Es = [ 2 2 2 2 4 5 7 8 3 10 11 3 13 14;
          1 3 4 7 5 6 8 9 10 11 12 13 14 15];
As = zeros(Ns , Ns);
for n=1:size(Es,2)
   As(Es(1,n), Es(2,n)) = 1;
   As(Es(2,n), Es(1,n)) = 1;
end
Ls = diag(As*ones(size(As,2),1))-As;
[Vs,Ds]=eig(Ls);

Nt = 2;
Et = [1:(Nt-1); 2:Nt];
At = zeros(Nt, Nt);
for n=1:size(Et,2)
   At(Et(1,n), Et(2,n)) = 1;
   At(Et(2,n), Et(1,n)) = 1;
end
Lt = diag(At*ones(size(At,2),1))-At;
[Vt,Dt]=eig(Lt);

% Take graph Cartesian product
Ast = kron(eye(Nt),As) + kron(At,eye(Ns));
Lst = diag(Ast*ones(size(Ast,2),1))-Ast;
[Vst,Dst]=eig(Lst);

% Plot
% initialize the skeleton
skeleton_loc = zeros(15,2);
skeleton_loc(1,:)=[0 2.5];
skeleton_loc(2,:)=[0 1.5]; 
skeleton_loc(3,:)=[0 0.5];
skeleton_loc(4,:)=[0.4 1.5];
skeleton_loc(5,:)=[0.8 1.1];
skeleton_loc(6,:)=[1.2 1.2];
skeleton_loc(7,:)=[-0.4 1.5];
skeleton_loc(8,:)=[-0.8 1.1];
skeleton_loc(9,:)=[-1.2 1.2];
skeleton_loc(10,:)=[0.3 0];
skeleton_loc(11,:)=[0.4 -0.85];
skeleton_loc(12,:)=[0.4 -1.75];
skeleton_loc(13,:)=[-0.3 0];
skeleton_loc(14,:)=[-0.4 -0.85];
skeleton_loc(15,:)=[-0.4 -1.75];

% Plot basis of Gs
bIdx = 2;
figure;
% first plot the fixed limbs
for m=1:size(Es,2)
    line( [skeleton_loc(Es(1,m),1), skeleton_loc(Es(2,m),1)],  [skeleton_loc(Es(1,m),2), skeleton_loc(Es(2,m),2)],...
    'LineStyle', '-', 'Color','k','LineWidth',2);
end
% plot all the joints
for m=1:Ns
    if Vs(m,bIdx)>0
        line(skeleton_loc(m,1),skeleton_loc(m,2),'marker','s', 'markersize',13,'MarkerFaceColor','b','color', 'b');
    else
        line(skeleton_loc(m,1),skeleton_loc(m,2),'marker','.', 'markersize',38,'MarkerFaceColor','r','color', 'r');
    end
end
axis off

% Plot basis of Gt
figure;
for bIdx=1:Nt
    subplot(2,1,bIdx);
    line( [0, 1],  [0, 0], 'LineStyle', '-', 'Color','k','LineWidth',2);
    m=1;
    if Vt(m,bIdx)>0
        line(0,0,'marker','s', 'markersize',13,'MarkerFaceColor','b','color', 'b');
    else
        line(0,0,'marker','.', 'markersize',38,'MarkerFaceColor','r','color', 'r');
    end
    m=2;
    if Vt(m,bIdx)>0
        line(1,0,'marker','s', 'markersize',13,'MarkerFaceColor','b','color', 'b');
    else
        line(1,0,'marker','.', 'markersize',38,'MarkerFaceColor','r','color', 'r');
    end
    axis off
end

% Plot basis of Gst
Vst_1 = kron(Vs(:,2), Vt(:,1)');
Vst_2 = kron(Vs(:,2), Vt(:,2)');
figure;
deltax = 3;
for n=1:Ns
    line( [skeleton_loc(n,1), skeleton_loc(n,1)+deltax],  [skeleton_loc(n,2), skeleton_loc(n,2)],...
    'LineStyle', '--', 'Color','m','LineWidth',1.5);
end
for m=1:size(Es,2)
    line( [skeleton_loc(Es(1,m),1)+deltax, skeleton_loc(Es(2,m),1)+deltax],  [skeleton_loc(Es(1,m),2), skeleton_loc(Es(2,m),2)],...
    'LineStyle', '-', 'Color','k','LineWidth',2);
end
for m=1:size(Es,2)
    line( [skeleton_loc(Es(1,m),1), skeleton_loc(Es(2,m),1)],  [skeleton_loc(Es(1,m),2), skeleton_loc(Es(2,m),2)],...
    'LineStyle', '-', 'Color','k','LineWidth',2);
end
for m=1:Ns
    if Vst_1(m)>0
        line(skeleton_loc(m,1),skeleton_loc(m,2),'marker','s', 'markersize',13,'MarkerFaceColor','b','color', 'b');
    else
        line(skeleton_loc(m,1),skeleton_loc(m,2),'marker','.', 'markersize',38,'MarkerFaceColor','r','color', 'r');
    end
end
for m=1:Ns
    if Vst_1(Ns+m)>0
        line(skeleton_loc(m,1)+deltax,skeleton_loc(m,2),'marker','s', 'markersize',13,'MarkerFaceColor','b','color', 'b');
    else
        line(skeleton_loc(m,1)+deltax,skeleton_loc(m,2),'marker','.', 'markersize',38,'MarkerFaceColor','r','color', 'r');
    end
end
axis([-1.5 4.5 -2 3])
axis off

figure;
for n=1:Ns
    line( [skeleton_loc(n,1), skeleton_loc(n,1)+deltax],  [skeleton_loc(n,2), skeleton_loc(n,2)],...
    'LineStyle', '--', 'Color','m','LineWidth',1.5);
end
for m=1:size(Es,2)
    line( [skeleton_loc(Es(1,m),1)+deltax, skeleton_loc(Es(2,m),1)+deltax],  [skeleton_loc(Es(1,m),2), skeleton_loc(Es(2,m),2)],...
    'LineStyle', '-', 'Color','k','LineWidth',2);
end
for m=1:size(Es,2)
    line( [skeleton_loc(Es(1,m),1), skeleton_loc(Es(2,m),1)],  [skeleton_loc(Es(1,m),2), skeleton_loc(Es(2,m),2)],...
    'LineStyle', '-', 'Color','k','LineWidth',2);
end
for m=1:Ns
    if Vst_2(m)>0
        line(skeleton_loc(m,1),skeleton_loc(m,2),'marker','s', 'markersize',13,'MarkerFaceColor','b','color', 'b');
    else
        line(skeleton_loc(m,1),skeleton_loc(m,2),'marker','.', 'markersize',38,'MarkerFaceColor','r','color', 'r');
    end
end
for m=1:Ns
    if Vst_2(Ns+m)>0
        line(skeleton_loc(m,1)+deltax,skeleton_loc(m,2),'marker','s', 'markersize',13,'MarkerFaceColor','b','color', 'b');
    else
        line(skeleton_loc(m,1)+deltax,skeleton_loc(m,2),'marker','.', 'markersize',38,'MarkerFaceColor','r','color', 'r');
    end
end
axis([-1.5 4.5 -2 3])
axis off


