// Clase de la partícula 
            class Particula {
                constructor(){
                		
                    this.x = Math.random()*anchura
					
                    this.x2 = this.x;
                    
                    this.y = Math.random()*altura
					
                    this.y2 = this.y;
					
                    var angleRadians = Math.atan2(altura/2 - this.y, anchura/2 - this.x) + Math.random()*Math.PI*2;
					
                    this.vx = (Math.cos(angleRadians+Math.PI/2))*1000
                    this.vy = (Math.sin(angleRadians+Math.PI/2))*1000
                    this.r = Math.round(Math.random()*255)
                    this.g = Math.round(Math.random()*255)
                    this.b = Math.round(Math.random()*255)
                    this.m = Math.random()*100+5
                }
            }
             // Clase de la partícula 
