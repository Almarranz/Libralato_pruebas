PRO ecu_to_gala

name='ACSWFC'
;~ name='WFC3IR'

cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'


readcol, cata + 'GALCEN_'+name+'_PM.cat',ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt,Format ='A,A,A,A,A,A,A,A,A,A,A,A';,SKIPLINE = 1

ra=float(ra)
dec=float(dec)
mua=float(mua)
mud=float(mud)


;~ mustar must be seted in order to get the same results as Libralato
;~ glactc_pm, ra, dec, mua, mud, 2000, gl, gb, mu_gl, mu_gb, 1, /DEGREE

;~ forprint, TEXTOUT= pruebas + name+'_ecu_to_gl_IDL.txt',gl, gb, mu_gl, mu_gb, COMMENT='gl, gb, mu_gl, mu_gb'

;~ ;Testing
;~ glactc_pm, 266.46036, -28.82440, -1.45, -2.68, 2000, gl, gb, mu_gl, mu_gb, 1, /DEGREE, /mustar
;~ print, mu_gl, mu_gb, 

openw,1, pruebas + name+'_ecu_to_gl_IDL.txt', 1

for i=0,n_elements(ra)-1 do begin
  
    glactc_pm, ra[i], dec[i], mua[i], mud[i], 2000, gl, gb, mu_gl, mu_gb, 1, /DEGREE
    glactc_pm, ra[i], dec[i], dmua[i], dmud[i], 2000, dgl, dgb, dmu_gl, dmu_gb, 1, /DEGREE
	
    openw, outp, pruebas + name+'_ecu_to_gl_IDL.txt', /get_lun, /APPEND
    printf, outp,format='(6(f, 6X))', gl, gb, mu_gl, mu_gb,dmu_gl, dmu_gb
    free_lun, outp

endfor


;~ readcol, pruebas + name+'_ecu_to_gl_IDL.txt',gl, gb, mu_gl, mu_gb,Format ='A,A,A,A'



END
