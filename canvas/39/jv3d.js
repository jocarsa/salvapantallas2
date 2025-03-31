var anchura = window.innerWidth;
           var altura = window.innerHeight;
            document.getElementById("lienzo").width = anchura;
            document.getElementById("lienzo").height = altura;
var contexto = document.getElementById("lienzo").getContext("2d");
            var centrox = document.getElementById("lienzo").width/2;
            var centroy = document.getElementById("lienzo").height/2;
            var lienzo = document.getElementById("lienzo");
            contexto.lineWidth=0
            /*
           contexto.shadowColor = "white";
contexto.shadowBlur = 5;
*/
            //contexto.shadowColor = "red";
//contexto.shadowBlur = 10;
			var r = Math.round(Math.random()*255)
                var g = Math.round(Math.random()*255)
                var b = Math.round(Math.random()*255)
            contexto.strokeStyle = "rgba("+r+","+g+","+b+",0.5)";
            contexto.fillStyle = 'rgba(0,0,255,0.01)'; 
var ratonPulsado = false;
 var ratonMedioPulsado = false;
            var mx = 0;
            var my = 0;

var V3d = function(x,y,z){
    this.x = parseFloat(x)*1;
    this.y = parseFloat(y)*1;
    this.z = parseFloat(z)*1;
}
var V2d = function(x,y){
    this.x = parseFloat(x)*1;
    this.y = parseFloat(y)*1;
}

var Cubo = function(centro,lado,color){
    var anchura = lado/2;
    this.color = color;
    this.zsort = 0;
     this.x = centro.x;
    this.y = centro.y;
    this.z = centro.z;
    this.vertices = [
        new V3d(centro.x-anchura, centro.y-anchura,centro.z + anchura),
        new V3d(centro.x-anchura, centro.y-anchura,centro.z - anchura),
        new V3d(centro.x+anchura, centro.y-anchura,centro.z - anchura),
        new V3d(centro.x+anchura, centro.y-anchura,centro.z + anchura),
        new V3d(centro.x+anchura, centro.y+anchura,centro.z + anchura),
        new V3d(centro.x+anchura, centro.y+anchura,centro.z - anchura),
        new V3d(centro.x-anchura, centro.y+anchura,centro.z - anchura),
        new V3d(centro.x-anchura, centro.y+anchura,centro.z + anchura)
    ];
    this.caras = [
        [this.vertices[0],this.vertices[1],this.vertices[2],this.vertices[3]],
        [this.vertices[4],this.vertices[5],this.vertices[6],this.vertices[7]],
        [this.vertices[3],this.vertices[2],this.vertices[5],this.vertices[4]],
        [this.vertices[7],this.vertices[6],this.vertices[1],this.vertices[0]],
        [this.vertices[7],this.vertices[0],this.vertices[3],this.vertices[4]],
        [this.vertices[1],this.vertices[6],this.vertices[5],this.vertices[2]]
    ];
    

}


var Caja = function(centro,dimensiones,color){
    
    this.color = color;
   
    this.vertices = [
        new V3d(centro.x-dimensiones.x, centro.y-dimensiones.y,centro.z + dimensiones.z),
        new V3d(centro.x-dimensiones.x, centro.y-dimensiones.y,centro.z - dimensiones.z),
        new V3d(centro.x+dimensiones.x, centro.y-dimensiones.y,centro.z - dimensiones.z),
        new V3d(centro.x+dimensiones.x, centro.y-dimensiones.y,centro.z + dimensiones.z),
        new V3d(centro.x+dimensiones.x, centro.y+dimensiones.y,centro.z + dimensiones.z),
        new V3d(centro.x+dimensiones.x, centro.y+dimensiones.y,centro.z - dimensiones.z),
        new V3d(centro.x-dimensiones.x, centro.y+dimensiones.y,centro.z - dimensiones.z),
        new V3d(centro.x-dimensiones.x, centro.y+dimensiones.y,centro.z + dimensiones.z)
    ];
    this.caras = [
        [this.vertices[0],this.vertices[1],this.vertices[2],this.vertices[3]],
        [this.vertices[4],this.vertices[5],this.vertices[6],this.vertices[7]],
        [this.vertices[3],this.vertices[2],this.vertices[5],this.vertices[4]],
        [this.vertices[7],this.vertices[6],this.vertices[1],this.vertices[0]],
        [this.vertices[7],this.vertices[0],this.vertices[3],this.vertices[4]],
        [this.vertices[1],this.vertices[6],this.vertices[5],this.vertices[2]]
    ];

}

var Cara3d = function(centro,puntos,color){
    this.centro = centro
    this.color = color;
    this.vertices = [
        new V3d(this.centro.x-puntos[0].x, this.centro.y-puntos[0].y,this.centro.z + puntos[0].z),
        new V3d(this.centro.x-puntos[1].x, this.centro.y-puntos[1].y,this.centro.z - puntos[1].z),
        new V3d(this.centro.x+puntos[2].x, this.centro.y-puntos[2].y,this.centro.z - puntos[2].z),
        new V3d(this.centro.x+puntos[3].x, this.centro.y-puntos[3].y,this.centro.z + puntos[3].z)
        
    ];
    this.caras = [
        [this.vertices[0],this.vertices[1],this.vertices[2],this.vertices[3]]
    ];

}

function proyeccion(O){
    var persp = 600;
    var calculo = persp/O.y;
    return new V2d(calculo*O.x,calculo*O.z);
}

function representacion(objetos,contexto,dx,dy){
    contexto.clearRect(0,0,anchura,altura); // Borro el canvas anterior
    objetos.sort((a, b) => {
	    return b.zsort - a.zsort;
	});
	for(var i = 0;i<objetos.length;i++){
            		//console.log(objetos[i].zsort)
            }
    
    for(var i = 0;i<objetos.length;i++){ // Para cada uno de los objetos
        contexto.fillStyle = objetos[i].color;
        if(objetos[i].vertices[0].y > 10){
	        for(var j = 0,ncaras = objetos[i].caras.length;j<ncaras;++j){
	            var cara = objetos[i].caras[j];
	            var P = proyeccion(cara[0]);
	            contexto.beginPath();
	            contexto.moveTo(P.x+dx,-P.y+dy);
	            for(var k=0,nvertices=cara.length;k<nvertices;++k){
	                P = proyeccion(cara[k]);
	                contexto.lineTo(P.x+dx,-P.y+dy);
	            }
	            contexto.closePath();
	            contexto.stroke();
	            contexto.fill();
	        }
        }else{
        		for(var j = 0,ncaras = objetos[i].vertices.length;j<ncaras;++j){
	            objetos[i].vertices[j].y += 300;
	            objetos[i].zsort += 300;
	        }
        }
    }
}
function representacionTransparencia(objetos,contexto,dx,dy,colordefondo){
    //contexto.clearRect(0,0,anchura,altura); // Borro el canvas anterior
    console.log(colordefondo)
    contexto.fillStyle = colordefondo; // Borro el canvas anterior
    contexto.fillRect(0,0,anchura,altura);
    objetos.sort((a, b) => {
	    return b.zsort - a.zsort;
	});
	
    for(var i = 0;i<objetos.length;i++){ // Para cada uno de los objetos
        contexto.fillStyle = objetos[i].color;
        contexto.strokeStyle = objetos[i].color;
        //contexto.shadowColor = objetos[i].color;
        if(objetos[i].vertices[0].y > 0){
	        for(var j = 0,ncaras = objetos[i].caras.length;j<ncaras;++j){
	            var cara = objetos[i].caras[j];
	            var P = proyeccion(cara[0]);
	            contexto.beginPath();
	            contexto.moveTo(P.x+dx,-P.y+dy);
	            for(var k=0,nvertices=cara.length;k<nvertices;++k){
	                P = proyeccion(cara[k]);
	                contexto.lineTo(P.x+dx,-P.y+dy);
	            }
	            
	            contexto.closePath();
	            contexto.stroke();
	            contexto.fill();
	        }
        }else{
        		for(var j = 0,ncaras = objetos[i].vertices.length;j<ncaras;++j){
	            objetos[i].vertices[j].y += 300;
	            objetos[i].zsort += 300;
	        }
        }
    }
}

function rotar(O,centro,theta,phi){
    var ct = Math.cos(theta);
    var st = Math.sin(theta);
    var cp = Math.cos(phi);
    var sp = Math.sin(phi);

    var x = O.x - centro.x;
    var y = O.y - centro.y;
    var z = O.z - centro.z;

    O.x = ct*x - st*cp*y + st*sp*z + centro.x;
    O.y = st*x + ct*cp*y - ct*sp*z + centro.y;
    O.z = sp*y + cp*z + centro.z
}
function empiezaMovimiento(event){
    ratonPulsado = true;
    mx = event.clientX;
    my = event.clientY;
    console.log(event.which)
    event.preventDefault();
}
function mover(event){
    if(ratonPulsado && event.which == 1){
        var theta = (event.clientX - mx)*Math.PI/360;
        var phi = (event.clientY - my)*Math.PI/180;
        for(var j = 0;j<cubos.length;j++){
        for(var i = 0;i<8;++i){
            rotar(cubos[j].vertices[i],centrocubo,theta,phi);
            mx = event.clientX;
            my = event.clientY;
            representacion(objetos,contexto,centrox,centroy);
            //console.log("hola")
        }
    }
        
        if(ratonPulsado && event.which == 2){
            
            centrox = centrox + (event.clientX - mx)*1;
            centroy = centroy + (event.clientY - my)*1;
            representacion(objetos,contexto,centrox,centroy);
            
        }

    }
}
function paraMovimiento(){
    ratonPulsado = false;
}

/*
lienzo.addEventListener('mousedown',empiezaMovimiento);
            document.addEventListener('mousemove',mover);
            document.addEventListener('mouseup',paraMovimiento)
            */