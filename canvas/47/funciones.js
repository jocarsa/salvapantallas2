// Calculo de la velocidad
            function calculaVelocidad(x1,y1,x2,y2,vx,vy,m1,m2){
					var angleRadians = Math.atan2(y2 - y1, x2 - x1);
					var distancia = Math.sqrt( Math.pow(x2 - x1,2) + Math.pow( y2 - y1,2) );
					vx += (Math.cos(angleRadians)/distancia)*(m2+1)
					vy += (Math.sin(angleRadians)/distancia)*(m2+1)
					return [vx,vy]
				}
