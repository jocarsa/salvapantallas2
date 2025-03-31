
<?php
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    if (isset($_FILES['image']) && $_FILES['image']['error'] === UPLOAD_ERR_OK) {
        // Directorio donde se guardará la imagen
        $uploadDir = 'imagenes/';
        if (!is_dir($uploadDir)) {
            mkdir($uploadDir, 0777, true); // Crear el directorio si no existe
        }
        
        // Nombre y ruta del archivo
        $uploadFile = $uploadDir . date('U').basename($_FILES['image']['name']);
        
        // Mueve el archivo subido al directorio especificado
        if (move_uploaded_file($_FILES['image']['tmp_name'], $uploadFile)) {
            echo "Imagen guardada exitosamente: " . $uploadFile;
        } else {
            echo "Error al guardar la imagen.";
        }
    } else {
        echo "No se recibió ninguna imagen o hubo un error en la subida.";
    }
} else {
    echo "Método no permitido.";
}
?>
<html>
  <body>
  <style>
    canvas{position:absolute;top:0px;left:0px;}
  </style>
  	<canvas id="lienzo2"></canvas>
    <canvas id="lienzo1"></canvas>
    
    <script>
      const lienzo1 = document.querySelector("#lienzo1")
      const lienzo2 = document.querySelector("#lienzo2")
      const contexto1 = lienzo1.getContext("2d")
      const contexto2 = lienzo2.getContext("2d")
      const anchura = window.innerWidth
      const altura = window.innerHeight
      
      
     
      lienzo1.width = anchura
      lienzo1.height = altura
      lienzo2.width = anchura
      lienzo2.height = altura
      
      let angulo1 = 0
      let radio1 = Math.random()*altura/4
      let velocidad1 = (Math.random()-0.5)/70
      
      let angulo2 = 0
      let radio2 = Math.random()*altura/4
      let velocidad2 = (Math.random()-0.5)/70
      
      let angulo3 = 0
      let radio3 = Math.random()*altura/4
      let velocidad3 = (Math.random()-0.5)/70
      
      
      
      let iniciox = anchura/2+Math.cos(angulo1)*radio1+Math.cos(angulo2)*radio2
      let inicioy = altura/2+Math.sin(angulo1)*radio1+Math.sin(angulo2)*radio2
      
      let contador = 0
      
      contexto2.fillStyle = "white"
      contexto2.fillRect(0,0,anchura,altura)
      contexto2.fillStyle = "black"
      
      function bucle(){
        contador++;
        contexto1.clearRect(0,0,anchura,altura)
         contexto1.lineWidth = 3
         // Circulo 0
         contexto1.beginPath()
         contexto1.arc(anchura/2,altura/2,5,0,Math.PI*2)
         contexto1.fill()
         
         // Brazo 1
        contexto1.beginPath()
        contexto1.moveTo(anchura/2,altura/2)
        contexto1.lineTo(anchura/2+Math.cos(angulo1)*radio1,altura/2+Math.sin(angulo1)*radio1)
        contexto1.stroke()
        // Circulo 1
         contexto1.beginPath()
         contexto1.arc(anchura/2+Math.cos(angulo1)*radio1,altura/2+Math.sin(angulo1)*radio1,5,0,Math.PI*2)
         contexto1.fill()
        // Brazo 2
        contexto1.beginPath()
        contexto1.moveTo(
          anchura/2+Math.cos(angulo1)*radio1,
          altura/2+Math.sin(angulo1)*radio1
        )
        contexto1.lineTo(
          anchura/2+Math.cos(angulo1)*radio1+Math.cos(angulo2)*radio2,
          altura/2+Math.sin(angulo1)*radio1+Math.sin(angulo2)*radio2
        )
        contexto1.stroke()
        // Circulo 2
        contexto1.beginPath()
         contexto1.arc(anchura/2+Math.cos(angulo1)*radio1+Math.cos(angulo2)*radio2,altura/2+Math.sin(angulo1)*radio1+Math.sin(angulo2)*radio2,5,0,Math.PI*2)
         contexto1.fill()
         
         // Brazo 3 ////////////////////////////////////////////////////////////
        contexto1.beginPath()
        contexto1.moveTo(
          anchura/2+Math.cos(angulo1)*radio1+Math.cos(angulo2)*radio2,
          altura/2+Math.sin(angulo1)*radio1+Math.sin(angulo2)*radio2
        )
        contexto1.lineTo(
          anchura/2+Math.cos(angulo1)*radio1+Math.cos(angulo2)*radio2+Math.cos(angulo3)*radio3,
          altura/2+Math.sin(angulo1)*radio1+Math.sin(angulo2)*radio2+Math.sin(angulo3)*radio3
        )
        contexto1.stroke()
        // Circulo 3
        contexto1.beginPath()
         contexto1.arc(
         anchura/2+Math.cos(angulo1)*radio1+Math.cos(angulo2)*radio2+Math.cos(angulo3)*radio3,
         altura/2+Math.sin(angulo1)*radio1+Math.sin(angulo2)*radio2+Math.sin(angulo3)*radio3
         ,5,0,Math.PI*2)
         contexto1.fill()
         
         
        // Circulo pintador
        contexto2.beginPath();
        contexto2.arc(
          anchura/2+Math.cos(angulo1)*radio1+Math.cos(angulo2)*radio2+Math.cos(angulo3)*radio3,
          altura/2+Math.sin(angulo1)*radio1+Math.sin(angulo2)*radio2+Math.sin(angulo3)*radio3,
          2,
          0,
          Math.PI*2
        )
        contexto2.fill()
        
        if(
          Math.abs(anchura/2+Math.cos(angulo1)*radio1+Math.cos(angulo2)*radio2+Math.cos(angulo3)*radio3 - iniciox) < 2
          &&
          Math.abs(altura/2+Math.sin(angulo1)*radio1+Math.sin(angulo2)*radio2+Math.sin(angulo3)*radio3 - inicioy) < 2
          &&
          contador > 1000
        ){
          enviarCanvas();
          
        }
        
        
        angulo1+=velocidad1
        angulo2+=velocidad2
        angulo3+=velocidad3
        requestAnimationFrame(bucle)
      }
      bucle()
      
      function enviarCanvas() {
        lienzo2.toBlob(function(blob) {
          var formData = new FormData();
          formData.append("image", blob, "canvas_image.jpg");

          fetch("?", {
            method: "POST",
            body: formData
          })
          .then(response => response.text())
          .then(function(datos){
          console.log(datos)
          	window.location = window.location
          } )
          .catch(error => console.error('Error:', error));
        }, "image/jpeg");
      }
      
    </script>
  </body>
</html>
