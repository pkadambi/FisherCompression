function hess_det=poly_reg(X, Y, Z, THRESHOLD)
    
    syms g(a,b);
    syms a, b;
    
    f = fit([X(:), Y(:)], Z, 'poly22', 'Exclude', Z>THRESHOLD );
    g(a,b) = f.p00 + f.p10*a+ f.p01*b+f.p20*a^2+f.p11 * a * b + f.p02 * b^2;
    poly22_hessian = vpa(hessian(g, [a,b]),3);
    hess_det = double(det(poly22_hessian));
    
end