<html>
	<head>
		<script src="https://code.jquery.com/jquery-3.6.1.js" integrity="sha256-3zlB5s2uwoUzrXK3BT7AX3FyvojsraNFxCc2vC/7pNI=" crossorigin="anonymous"></script>
		<style>
			
			html,body,#contenedor{
				background:black;
				color:white;
				width:100%;
				height:100%;
				overflow:hidden;
				color:green;
			}
			.osui{
				width:400px;
				height:400px;
				border:1px solid white;
				border-top:20px solid white;
				border-radius:5px;
				position:absolute;
				backdrop-filter: blur(10px);
				color:green;
				box-shadow:0px 10px 20px black;
			}
			iframe{
				width:400px;
				height:400px;
				border:0px;
				outline:none;
				color:green;
				
			}
			#inlineFrameExample{
				box-shadow:0px 5px 10px rgba(255,255,255,0.3);
			}
			#matrix{
				width:100%;
				height:100%;
				position:absolute;
			}
		</style>
	</head>
	<body>
		<iframe id="matrix"
						    title="Inline Frame Example"
						    width="400"
						    height="400"
						    src="https://jocarsa.com/go/matrix/index">
						</iframe>
		<div id="contenedor"></div>
		<script>
		var anchura = window.innerWidth
		var altura = window.innerHeight
		var tiempo = 0;
		var temporizador = setTimeout("bucle()",1000)
			function bucle(){
				tiempo++;
				if(Math.random()<0.01){
					var new_item = $(`
						<div class="osui" style="left:`+(Math.random()*(anchura-400))+`px;top:`+(Math.random()*(altura-400))+`;display:none;">
						<iframe id="inlineFrameExample"
						    title="Inline Frame Example"
						    width="`+(Math.random()*(400)+200)+`"
						    height="`+(Math.random()*(400)+200)+`"
						    
						    src="https://jocarsa.com/go/salvapantallas/28/index">
						</iframe>
						</div>
						`).hide();
					nuevocontenedor = $("#contenedor").append(new_item)
					new_item.show('slow');
					
				}
				if(Math.random()<0.002){
					things = $('.osui');
					$(things[Math.floor(Math.random()*things.length)]).hide("puff").delay(10).queue(function(){$(this).remove();});
					
				}
				if($('.osui').length > 10){
					things = $('.osui');
					$(things[Math.floor(Math.random()*things.length)]).remove()
				}
				clearTimeout(temporizador)
				temporizador = setTimeout("bucle()",10)
			}
		</script>
	</body>
</html>