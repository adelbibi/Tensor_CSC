function d_disp=plotfilters(d,iter)

pd=1;
sqr_k=ceil(sqrt(size(d,3)));
filter_radius=floor(size(d,1)/2);
d_disp = zeros( sqr_k * [filter_radius*2+1 + pd, filter_radius*2+1 + pd] + [pd, pd]);
for j = 0:size(d,3)-1
d_disp( floor(j/sqr_k) * (size(d,1) + pd) + pd + (1:size(d,1)) , mod(j,sqr_k) * (size(d,2) + pd) + pd + (1:size(d,2)) ) = d(:,:,j+1);
end
imagesc(d_disp), axis off, colormap gray;

end