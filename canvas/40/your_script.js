// Set up the scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Create custom shader material
const vertexShader = `
    void main() {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

const fragmentShader = `
    uniform vec2 resolution;
    uniform float time;
    
    void main() {
        vec2 p = (2.0 * gl_FragCoord.xy - resolution) / min(resolution.x, resolution.y);
        p *= 10.0;
        float d = mod(length(p), 1.0);
        gl_FragColor = vec4(vec3(d), 1.0);
    }
`;

const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
        resolution: { value: new THREE.Vector2() },
        time: { value: 0 },
    },
});

// Create a plane and add it to the scene
const geometry = new THREE.PlaneGeometry(2, 2);
const plane = new THREE.Mesh(geometry, material);
scene.add(plane);

// Set camera position
camera.position.z = 1;

// Create an animation loop
const animate = (time) => {
    material.uniforms.time.value = time * 0.001;
    material.uniforms.resolution.value.set(renderer.domElement.width, renderer.domElement.height);
    
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
};

// Handle window resize
window.addEventListener('resize', () => {
    const newWidth = window.innerWidth;
    const newHeight = window.innerHeight;

    camera.aspect = newWidth / newHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(newWidth, newHeight);
});

// Start the animation loop
animate(0);
