function a_n=normalized(a)

a_n=(a-min(a(:)))./(max(a(:))-min(a(:)));

end