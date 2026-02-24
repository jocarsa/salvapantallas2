<html>
    <head>
        <script src="https://code.jquery.com/jquery-3.6.1.js" integrity="sha256-3zlB5s2uwoUzrXK3BT7AX3FyvojsraNFxCc2vC/7pNI=" crossorigin="anonymous"></script>
        <style>
            @font-face {
              font-family: mifuente;
              src: url(computer_7.ttf);
            }
            body{color:white;text-shadow:0px 0px 2px white;font-family:monospace;overflow:hidden;font-size:8px;}
        </style>
    </head>
    <body id="cuerpo">
    
    </body>
    <script>
        codigo = `<?php 
                $archivos = scandir("codigo");
                array_splice($archivos, 0, 2);

                $numero = rand(0,count($archivos)-1);
                    //echo "codigo/".$archivos[$numero];
                    include "codigo/".str_replace("`","'",$archivos[$numero]);
                    
            ?>`;
        var contador = 0;
        var temporizador = setTimeout("bucle()",1000)
        var reg = /\t/;
        function bucle(){
            if(codigo[contador] == "\n"){
                $("body").append("<br>")
            }else if(codigo[contador].search(reg) == true){
                   $("body").append("----")     
                     }else{
                $("body").append(codigo[contador])
            }
            $("html, body").animate({ scrollTop: 1000000000 });
            contador++;
            if(contador == codigo.length){window.location = window.location}
            clearTimeout(temporizador)
            temporizador = setTimeout("bucle()",Math.round(Math.random()*20)+1)
        }
    </script>
</html>