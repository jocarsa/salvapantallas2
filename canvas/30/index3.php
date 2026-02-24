<html>
	<head>
		<script src="https://code.jquery.com/jquery-3.6.1.js" integrity="sha256-3zlB5s2uwoUzrXK3BT7AX3FyvojsraNFxCc2vC/7pNI=" crossorigin="anonymous"></script>
		<style>
			
			html,body,#contenedor{
				background:white;
				color:black;
				width:100%;
				height:100%;
				/*overflow:hidden;*/
				filter:invert(100%);
			}
			.osui{
				width:400px;
				height:400px;
				
				position:absolute;
				
			}
			iframe{
				width:400px;
				height:400px;
				border:0px;
				outline:none;
				
			}
			
			#matrix{
				width:100%;
				height:100%;
				position:absolute;
				opacity:0.1;
				
			}
			#overlay{
				position:absolute;
				top:0px;
				left:0px;
				width:100%;
				height:100%;
				background: rgb(2,0,36);

opacity:0.2;
<?php echo 'filter:hue-rotate('.rand(0,360).'deg);';?>
			}
		</style>
	</head>
	<body>
		
		<div id="contenedor"></div>
		<div id="overlay"></div>
		<script>
		var anchura = window.innerWidth
		var altura = window.innerHeight
		var tiempo = 0;
		var temporizador = setTimeout("bucle()",1000)
			function bucle(){
				tiempo++;
				if(Math.random()<0.01){
					nuevocontenedor = $("#contenedor").append(`
						<div class="osui" style="left:`+(Math.random()*(anchura-400))+`px;top:`+(Math.random()*(altura-400))+`;filter:blur(`+Math.round(Math.random()*10)+`px)">
						<iframe id="inlineFrameExample"
						    title="Inline Frame Example"
						    width="400"
						    height="400"
						    
						    src="https://jocarsa.com/go/salvapantallas/28/index">
						</iframe>
						</div>
						`)
					
				}
				if(Math.random()<0.002){
					things = $('.osui');
					$(things[Math.floor(Math.random()*things.length)]).remove()
					
				}
				if($('.osui').length > 20){
					things = $('.osui');
					$(things[Math.floor(Math.random()*things.length)]).remove()
				}
				clearTimeout(temporizador)
				temporizador = setTimeout("bucle()",10)
			}
		</script>
	</body>
</html>