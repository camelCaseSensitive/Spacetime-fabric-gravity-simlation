const vert = `
uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
uniform float uStrokeWeight;

uniform float mouseX;
uniform float mouseY;
uniform float a;
uniform float s;


uniform vec4 uViewport;
uniform int uPerspective;

attribute vec4 aPosition;
attribute vec4 aDirection;

vec4 offset(vec4 inPosition) {
  return vec4(inPosition.x, a*exp(-1.0*((inPosition.x-mouseX)*(inPosition.x-mouseX) / (s*s*2.0) + (inPosition.y-mouseY)*(inPosition.y-mouseY) / (s*s*2.0)))-25.0/400.0, inPosition.y, 1.);
}

  
void main() {
  
  vec3 scale = vec3(0.9995);

  vec4 add = vec4(aPosition.x, a*exp(-1.0*((aPosition.x-mouseX)*(aPosition.x-mouseX) / (s*s*2.0) + (aPosition.y-mouseY)*(aPosition.y-mouseY) / (s*s*2.0)))-25.0/400.0, aPosition.y, 1.);

  vec4 newPos = add;

  vec4 posp = uModelViewMatrix * offset(aPosition);
  vec4 posq = uModelViewMatrix * offset(aPosition + vec4(aDirection.xyz, 0));

  posp.xyz = posp.xyz * scale;
  posq.xyz = posq.xyz * scale;

  vec4 p = uProjectionMatrix * posp;
  vec4 q = uProjectionMatrix * posq;

  vec2 tangent = normalize((q.xy*p.w - p.xy*q.w) * uViewport.zw);

  // flip tangent to normal (it's already normalized)
  vec2 normal = vec2(-tangent.y, tangent.x);

  float thickness = aDirection.w * uStrokeWeight;
  vec2 offset = normal * thickness / 2.0;

  vec2 curPerspScale;

  if(uPerspective == 1) {
    curPerspScale = (uProjectionMatrix * vec4(1, -1, 0, 0)).xy;
  } else {
    curPerspScale = p.w / (0.5 * uViewport.zw);
  }
  gl_Position.xy = p.xy + offset.xy * curPerspScale;
  gl_Position.zw = p.zw;
}
`

const frag = `
precision mediump float;
precision mediump int;

uniform vec4 uMaterialColor;

void main() {
  gl_FragColor = vec4(255., 255., 255., 1.);
}
`

const makePlane = (detail) => {
  return new p5.Geometry(detail, detail, function() {

    const [uVec, vVec] = [createVector(1, 0, 0), createVector(0, 1, 0)]
    const normal = uVec.cross(vVec)
  
    // This will be the index of the first vertex
    // of this face
    const vertexOffset = this.vertices.length

    for (let i = 0; i < detail; i++) {
      for (let j = 0; j < detail; j++) {
        const u = i / (detail - 1)
        const v = j / (detail - 1)
        this.vertices.push(
          createVector(0, 0, 0)
          .add(uVec.copy().mult(u - 0.5))
          .add(vVec.copy().mult(v - 0.5))
        )
      }
    }
    this.strokeIndices = []
    for (let i = 1; i < detail; i++) {
      for (let j = 1; j < detail; j++) {
        if(j%5 == 0) {this.strokeIndices.push([
          vertexOffset + (j - 1) * detail + i - 1,
          vertexOffset + (j - 1) * detail + i,
        ])
        }
        if(i%5 == 0){
          this.strokeIndices.push([
            vertexOffset + (j - 1) * detail + i,
            vertexOffset + j * detail + i,
          ])
        }
      }
    }
  })
}

function setup() {
  p5.disableFriendlyErrors = true; // disables FES
  // WEBGL is a bit trick with 3D beacuse the camera is oriented with the y-axis down and a left handed coordinate system
  // For this reason we are drawing the space-time fabric on the X-Z plane.  So think of the negative y-axis as UP!
  createCanvas(400, 400, WEBGL);
  distortShader = createShader(vert, frag)
  gridPlane = makePlane(150)
  cam = createCamera();
  frameRate(20)
  
  // Focal Plane is a 1/10 scaled widthxheight rectangle, 35 away from camera center
  // By passing the argument 350/10 I have set the focal plane to be 35 units away from the camera center
  perspective(PI/3, width/height, 350/10, 350*10)   
  
  // ********** Starting camera position ***************
  // Set the starting camera position to whatever you like and then call cam.lookAt(0,0,0) to orient the view to the x-z plane
  // Turn on orbit control and console.log([cam.eyeX, cam.eyeY, cam.eyeZ]) in the mousePressed() function to see where your cam is positioned
  // Then plus those desired values into the cam.setPosition() function below.
  cam.setPosition(0, -250, -400)
  cam.lookAt(0,0,0)
  
  // Initial pan (Ry) and tilt (Rx) angles of the camera
  thY = atan2(cam.eyeX, cam.eyeZ);
  thX = atan2(cam.eyeY, sqrt(cam.eyeX**2 + cam.eyeZ**2))

  // Rotation from screen coordinates to world coordinates
  Rrev = matrixMult(Rx(thX), Ry(-thY))

  // mouseVector is the vector from the camera center to where the users mouse intersects the focal plane
  // mouseVector = [[(mouseX - width/2)/10, (mouseY - height/2)/10, -35, 1]]
  mouseVector = [[0, 0, -35, 1]]

  // We then make this a unit vector (a vector with magnitude = 1)
  mouseVector = vecNorm(mouseVector)
  // We next define the mouseVector in the world coordinates by multiplying by the rotation matrix
  worldVector = matrixMult(mouseVector, Rrev)

  // We then calculate the value which we need to multiply the y-component of this unit vector by
  // such that the y-component will be equal to the camera's y position.
  // Basically - when we multiply our unit vector by "t" it will intersect exactly with the x-z plane
  t = -cam.eyeY/worldVector[0][1]
  //  so then (cam.eyeX+worldVector[0][0]*t, cam.eyeY+worldVector[0][1]*t, cam.eyeZ+worldVector[0][2]*t)
  // the above is then the location we would draw our sphere so that it tracks the mouse.

  pColors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

  startRadius = 500; // Radius from the center from which the small planets fly in from

  gTh = 0;    // The angle between the small planet and the center gravitational planet
  gForce = 0; // The force of gravity - initialized here - calculated each frame for each planet since it varied due to distance
  gFactor = 1;// The gravitational force multiplier which scales with the center planet's radius (bigRad)
  gRange = 1.1*startRadius    // Radius from center planet at which gravity no longer has an effect (controlled by an if statement)

  mercury = new Planet({ p: {x: -300, y: -25, z: -300}, v: {x: 1, y: 0, z: 1}, a: {x: 0, z: 0}, r: 7, th: 0, speed: 30, d: 10000})
  mercury.initialize();
  
  planets = [mercury]

  pathRes = 1;
  pathCount = 0;

  bigRadInitial = 24      // This is used to calculate the gFactor later on
  bigRad = bigRadInitial; // Radius of the large center planet

  // These values control the shape of the space-time warp - they increase as the planet gains mass
  sd = 45;    // Standard deviation of the gaussian curve - corresponds to how wide it is 
  amp = 50;   // Amplitude of the gaussian curve - corresponds to how deep it is
}

class Planet {
  constructor(params){
    Object.assign(this, params);
    this.gForce = 0;
    this.gTh = 0;
    this.color = pColors[round(random(0, pColors.length - 1))];
    this.speedCap = 35;
    this.trail = [];
  }

  initialize() {
    // Initialize start location to a random point on a r=startRadius (set to 500) circle around the origin
    this.p.x = random(-startRadius , startRadius);
    this.p.y = -25;
    this.p.z = sign(random(-1,1))*sqrt(startRadius**2 - this.p.x**2)
    if(gFactor < 5) {
      this.th = atan2(this.p.z + (cam.eyeZ+worldVector[0][2]*t), this.p.x + (cam.eyeX+worldVector[0][0]*t)) + sign(random(-1,1)) * (PI/8 + gFactor/5 * (PI/2 - PI/8));
    } else {
      this.th = atan2(this.p.z + (cam.eyeZ+worldVector[0][2]*t), this.p.x + (cam.eyeX+worldVector[0][0]*t)) + sign(random(-1,1)) * (PI/2);
    }
    // console.log(this.th)
    this.v.x = -this.speed*cos(this.th);
    this.v.z = -this.speed*sin(this.th);
    this.r = 7 + random(-2, 6)
    this.color = pColors[round(random(0, pColors.length - 1))]
    // A low max speed (speedCap) for the small planets prevents them from slingshotting through the center planet's gravitational field and missing
    // A high speedCap allows planets to slingshot out of gravitational range and miss
    // this.speedCap = 10000;  // Randomize to have more or fewer planets miss
    this.speedCap = 35;
    this.trail = [];  // planet trail see line 208
  }

  update() {
      // Update velcity
      this.v.x += this.a.x;
      this.v.z += this.a.z;

      // Update the planet's heading (velocity angle)
      this.th = atan2(this.v.z, this.v.x);

      // vel = speed AKA velocity magnitude
      let vel = sqrt(this.v.x**2 + this.v.z**2)

      // Keep the planet below the speedCap (see initialization method above for more info)
      if(vel > this.speedCap){
        this.v.x = this.speedCap*cos(this.th)
        this.v.z = this.speedCap*sin(this.th)
      }

      // Cause small planet to lose energy when it is within gRange so that it will spiral to center
      if(this.d < gRange){
        this.v.x *= 0.98;
        this.v.z *= 0.98;
      }
      
      // Update the position
      this.p.x += this.v.x;
      this.p.z += this.v.z;
  }

  draw() {
    // Draw the small planet
    noStroke()
    fill(this.color)
    sphereAt(this.p.x, this.p.y, this.p.z, this.r)

    // Draw the path
    let path = this.trail;
    // stroke(this.color)
    if(path.length > 2){
      for(let i = 0; i < path.length-1; i++){
        sphereAt(path[i][0], path[i][1], path[i][2], 3 * i/(path.length-1))
        // drawLine(path[i][0], path[i][1], path[i][2], path[i+1][0], path[i+1][1], path[i+1][2])
      }
    }
  }
}

function draw() {
  orbitControl(3)
  lights()
  background(0);
  noFill()
  
  pointLight(255, 255, 255, 25, -255, 15)

  // 3D Coordinates
  // stroke(255,0,0)
  // drawLine(0,0,0, 150,0,0)
  // stroke(0,255,0)
  // drawLine(0,0,0, 0,150,0)
  // stroke(0,0,255)
  // drawLine(0,0,0, 0,0,150)
  // stroke(0);

  thY = atan2(cam.eyeX, cam.eyeZ);
  thX = atan2(cam.eyeY, sqrt(cam.eyeX**2 + cam.eyeZ**2))
  Rrev = matrixMult(Rx(thX), Ry(-thY))
  if(mouseX > 0 && mouseX < width && mouseY > 0 && mouseY < height) {
    mouseVector = [[(mouseX - width/2)/10, (mouseY - height/2)/10, -35, 1]]
  } else {
    mouseVector = [[0, 0, -35, 1]]
  }
  
  mouseVector = vecNorm(mouseVector)
  worldVector = matrixMult(mouseVector, Rrev)

  t = -cam.eyeY/worldVector[0][1]

  const scl = 400
  push()
  shader(distortShader)
  distortShader.setUniform('mouseX', (cam.eyeX+worldVector[0][0]*t)/scl)
  distortShader.setUniform('mouseY', (cam.eyeZ+worldVector[0][2]*t)/scl)
  distortShader.setUniform('a', amp/scl);
  distortShader.setUniform("s", sd/scl)
  scale(scl)
  noFill()
  stroke(255)
  strokeWeight(2)
  model(gridPlane)
  pop()
  // resetShader()

  
  noStroke();
  fill('#FFCC00')
  sphereAt(cam.eyeX+worldVector[0][0]*t, cam.eyeY+worldVector[0][1]*t + amp-25 - bigRad, cam.eyeZ+worldVector[0][2]*t, bigRad)
  stroke(0)

  // Gaussian lines
  // for(let i = -175; i < 175; i+=35){
  //   graphXYCurve(gaussian, i, 175, 10, amp, cam.eyeX+worldVector[0][0]*t, cam.eyeZ+worldVector[0][2]*t, sd)
  //   graphZYCurve(gaussian, i, 175, 10, amp, cam.eyeX+worldVector[0][0]*t, cam.eyeZ+worldVector[0][2]*t, sd)
  // };
  for(let i = 0; i < planets.length; i++){
    let body = planets[i]
    body.d = dist3D(body.p.x, body.p.y, body.p.z, cam.eyeX+worldVector[0][0]*t, cam.eyeY+worldVector[0][1]*t + amp-50, cam.eyeZ+worldVector[0][2]*t)
    gTh = atan2( (cam.eyeZ+worldVector[0][2]*t) - body.p.z, (cam.eyeX+worldVector[0][0]*t) - body.p.x)
    gForce = 5000 * gFactor / (body.d**1.5)
    if(body.d > gRange) gForce = 0;
    body.a.x = gForce * cos(gTh);
    body.a.z = gForce * sin(gTh);
    body.update();
    // drawLine(body.p.x, body.p.y, body.p.z, cam.eyeX+worldVector[0][0]*t, cam.eyeY+worldVector[0][1]*t, cam.eyeZ+worldVector[0][2]*t)
    
    // *** Collision Detection ** 
    if (body.d < bigRad + 5){
      body.initialize();
      // Increasing the standard deviation and amplitude of the Gaussian makes the space-time warp wider and deeper, respectively
      sd += 2;
      amp +=5;
      // Changing bigRad makes the center planet larger in radius
      bigRad += 1;
      // Note: the ratio of the sd, amp and bigRad values is important to prevent the planet from growing out of the mesh
      // I would recommend keeping the ratio roughly proportional to what it is now
      // The main thing is to ensure the sd (~Guassian width) is larger than the bigRad increment 

      // The gFactor increases with bigRad - increase or decrease the denominator (10) to decrease or increase, respectively, the gForce increment
      gFactor = 1 + (bigRad-bigRadInitial)/10;
    } else if (body.d > 800) {
      // If body misses and is now too far away reinitialize
      body.initialize();
    }
    // Set the y position of the body to the value of the Gaussian at the body's current position
    // This keeps the body on the surface of the grid as it sprials down to the large center planet
    body.p.y = gaussian(body.p.x, body.p.z, amp, cam.eyeX+worldVector[0][0]*t, cam.eyeZ+worldVector[0][2]*t, sd) - body.r;
    if(pathCount % pathRes == 0) {
      body.trail.push([body.p.x, body.p.y, body.p.z])
    }
    pathCount ++;
    body.draw();
  }

   // Comment out to fix the view for the user
}

// Returns the sign of a given value 'n' 
// NOTE: Function cannot return 0 because this function is used only to set the random planet location and I need a -1 or 1
function sign(n) {
  if(n > 0){
    return 1;
  } else {
    return -1;
  }
}

function dist3D(x1, y1, z1, x2, y2, z2) {
  return sqrt( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )
}

// function thBetween(u, v) {
//   let uMag = sqrt(u[0]**2 + u[1]**2)
//   let vMag = sqrt(v[0]**2 + v[1]**2)

//   return acos( dot(u,v) / (uMag * vMag))
// }

function mousePressed() {
  // *** Log camera position ***
  // console.log([cam.eyeX, cam.eyeY, cam.eyeZ])

  // let p = new Planet({ p: {x: -300, y: -25, z: -300}, v: {x: 1, y: 0, z: 1}, a: {x: 0, z: 0}, r: 7, th: 0, speed: 30, d: 10000})
  // p.initialize();
  // planets.push(p)
  console.log(frameRate())

  // planets[0].initialize();
  // console.log(sd)
}

function drawLine(x1, y1, z1, x2,y2, z2){
  beginShape();
  vertex(x1,y1,z1);
  vertex(x2,y2,z2);  
  endShape();
}

function sphereAt(a, b, c, rad){
  push();
  translate(a, b, c)
  sphere(rad)
  pop()
}

// The function passed to these functions (f) is the Gaussian
// function graphXZCurve(f, y, domain, i, a, xo, yo, s){
//   for(let x = -domain; x < domain; x+=i){
//     drawLine(x, y, f(x, y, a, xo, yo, s), x+i, y, f(x+i, y, a, xo, yo, s));
//   }
// }

// function graphYZCurve(f, x, domain, i, a, xo, yo, s){
//   for(let y = -domain; y < domain; y+=i){
//     drawLine(x, y, f(x, y, a, xo, yo, s), x, y+i, f(x, y+i, a, xo, yo, s));
//   }
// }

function graphXYCurve(f, z, domain, i, a, xo, yo, s){
  for(let x = -domain; x < domain; x+=i){
    // drawLine(x, f(x, z, a, xo, yo, s), z, x+i, f(x+i, z, a, xo, yo, s), z);
    // sphereAt(x, f(x, z, a, xo, yo, s), z, 0.5)
    line(x, f(x, z, a, xo, yo, s), z, x+i, f(x+i, z, a, xo, yo, s), z);
  }
}

function graphZYCurve(f, x, domain, i, a, xo, yo, s){
  for(let z = -domain; z < domain; z+=i){
    // drawLine(x, f(x, z, a, xo, yo, s), z, x, f(x, z+i, a, xo, yo, s), z+i);
    // sphereAt(x, f(x, z, a, xo, yo, s), z, 0.5)
    line(x, f(x, z, a, xo, yo, s), z, x, f(x, z+i, a, xo, yo, s), z+i);
  }
}

// Gaussian Function - returns the z-value of a 2D Gaussian Function at a point (x, y) 
// given the center (xo, yo), height (a) and standard deviation (s)
function gaussian(x, y, a, xo, yo, s) {
  return a*exp(-((x-xo)**2 / (s**2 *2) + (y-yo)**2 / (s**2 *2))) - 25;
}

// Rotation matrices - used for transforming screen coordinates to coordinates on the X-Z plane
// Rx rotates by 'th' about the x-axis, Ry about the y-axis, Rz about the z-axis
function Rx(th) {
  return [[1, 0, 0, 0],
        [0, cos(th), -sin(th), 0],
        [0, sin(th), cos(th), 0],
        [0, 0, 0, 1]]
} 

function Ry(th) {
  return [[cos(th), 0, sin(th), 0],
        [0, 1, 0, 0],
        [-sin(th), 0, cos(th), 0],
        [0, 0, 0, 1]]
} 

function Rz(th) {
  return [[cos(th), -sin(th), 0, 0],
        [sin(th), cos(th), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
} 

function matrixMult(A, B) {
  if(A[0].length !== B.length) return "A col != B row"
  l = A.length;      // Number of rows in A
  m = A[0].length;   // Number of columns in A and number of rows in B
  n = B[0].length;   // Number of columns in B
  
  // console.log("A is an :" + l + "x" + m + " Matrix ")
  // console.log("B is an :" + m + "x" + n + " Matrix ")
  
  let C = []
  for(let i = 0; i < l; i++){
    C[i] = [];
    for(let j = 0; j < n; j++){
      C[i][j] = [];
    }
  }
  
  for(let row = 0; row < l ; row++){
    for(let col = 0; col < n; col++){
      let v = [];
      let w = [];
      for(let i = 0; i < m ; i++){
         v.push(A[row][i])
         w.push(B[i][col])
      }
      C[row][col] = dot(v,w)
    }
  }
  return C;
}

function dot(v, w){
  if(v.length != w.length) return "Error, vector lengths do not match"
  let sum = 0;
  for(i = 0; i < v.length; i++){
    sum += v[i] * w[i];
  }
  return sum;
}

// Vector magnitude of a two vectors [[x, y, z, 1]]
function vecNorm(v) {
  let vmag = sqrt(v[0][0]**2 + v[0][1]**2 + v[0][2]**2)
  return [[v[0][0] / vmag, v[0][1] / vmag, v[0][2] / vmag, 1]]
}

