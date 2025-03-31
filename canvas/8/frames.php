<?php header("Origin-Agent-Cluster: ?1"); ?>
<html>
    <head>
        <style>
            body,html{padding:0px;margin:0px;overflow:hidden;}
            iframe{border:0px;position:absolute;top:0px;left:0px;mix-blend-mode:lighten;}
            
        </style>
        
    </head>
    <body>
        <iframe src="index.html" width=1920 height=1080 id="frame1"></iframe>
        <iframe src="index.html" width=1920 height=1080 id="frame2"></iframe>
        <iframe src="index.html" width=1920 height=1080 id="frame3"></iframe>
        <iframe src="index.html" width=1920 height=1080 id="frame4"></iframe>
        <iframe src="index.html" width=1920 height=1080 id="frame5"></iframe>
        <iframe src="index.html" width=1920 height=1080 id="frame6"></iframe>
        <iframe src="index.html" width=1920 height=1080 id="frame7"></iframe>
        <iframe src="index.html" width=1920 height=1080 id="frame8"></iframe>
        
        <script>
            var anchura = window.innerWidth;
            var altura = window.innerHeight;
            document.getElementById("frame1").width=anchura;
            document.getElementById("frame1").height=altura;
            
            document.getElementById("frame2").width=anchura;
            document.getElementById("frame2").height=altura;
            
            document.getElementById("frame3").width=anchura;
            document.getElementById("frame3").height=altura;
            
            document.getElementById("frame4").width=anchura;
            document.getElementById("frame4").height=altura;
            
            document.getElementById("frame5").width=anchura;
            document.getElementById("frame5").height=altura;
            
            document.getElementById("frame6").width=anchura;
            document.getElementById("frame6").height=altura;
            
            document.getElementById("frame7").width=anchura;
            document.getElementById("frame7").height=altura;
            
            document.getElementById("frame8").width=anchura;
            document.getElementById("frame8").height=altura;
            
        </script>
    </body>
</html>