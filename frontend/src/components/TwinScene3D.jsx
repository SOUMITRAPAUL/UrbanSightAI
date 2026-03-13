import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

function collectBounds(data) {
  const points = []
  const push = (lat, lon) => {
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      points.push([lat, lon])
    }
  }

  ;(data?.boundary ?? []).forEach(([lat, lon]) => push(lat, lon))
  ;(data?.layers?.roads ?? []).forEach((line) => line.forEach(([lat, lon]) => push(lat, lon)))
  ;(data?.layers?.drains ?? []).forEach((line) => line.forEach(([lat, lon]) => push(lat, lon)))
  ;(data?.layers?.rivers ?? []).forEach((line) => line.forEach(([lat, lon]) => push(lat, lon)))
  ;(data?.layers?.waterbodies ?? []).forEach((line) =>
    line.forEach(([lat, lon]) => push(lat, lon))
  )
  ;(data?.layers?.blocked_drain_network ?? []).forEach((segment) =>
    (segment?.path ?? []).forEach(([lat, lon]) => push(lat, lon))
  )
  ;(data?.layers?.houses ?? []).forEach((item) => push(item.lat, item.lon))
  ;(data?.layers?.parks ?? []).forEach((item) => push(item.lat, item.lon))
  ;(data?.layers?.playgrounds ?? []).forEach((item) => push(item.lat, item.lon))
  ;(data?.layers?.blocked_drains ?? []).forEach((item) => push(item.lat, item.lon))
  ;(data?.layers?.flood_zones ?? []).forEach((item) => push(item.lat, item.lon))
  ;(data?.hotspots ?? []).forEach((item) => push(item.lat, item.lon))

  if (!points.length) {
    return { latMin: 0, latMax: 1, lonMin: 0, lonMax: 1 }
  }
  const lats = points.map((p) => p[0])
  const lons = points.map((p) => p[1])
  return {
    latMin: Math.min(...lats),
    latMax: Math.max(...lats),
    lonMin: Math.min(...lons),
    lonMax: Math.max(...lons),
  }
}

function toWorld(lat, lon, bounds) {
  const sx = (lon - bounds.lonMin) / (bounds.lonMax - bounds.lonMin + 1e-8)
  const sz = (lat - bounds.latMin) / (bounds.latMax - bounds.latMin + 1e-8)
  const x = (sx - 0.5) * 160
  const z = (sz - 0.5) * 120
  return new THREE.Vector3(x, 0, z)
}

function pathToWorldPoints(path, bounds, y) {
  if (!Array.isArray(path)) return []
  return path
    .filter((pt) => Array.isArray(pt) && pt.length === 2)
    .map(([lat, lon]) => {
      const world = toWorld(lat, lon, bounds)
      return new THREE.Vector3(world.x, y, world.z)
    })
}

function toWorldRadiusMeters(radiusMeters, bounds) {
  const latSpan = Math.max(bounds.latMax - bounds.latMin, 0.001)
  const lonSpan = Math.max(bounds.lonMax - bounds.lonMin, 0.001)
  const avgSpanKm = ((latSpan + lonSpan) / 2) * 111
  const normalized = (radiusMeters / 1000) / Math.max(avgSpanKm, 0.1)
  return Math.max(2.5, normalized * 140)
}

function severityColor(severity) {
  if (severity === 'critical') return 0xf95a5a
  if (severity === 'high') return 0xff923d
  if (severity === 'medium') return 0xffd85e
  return 0x73d8ac
}

function severityLabel(severity, riskScore = null) {
  if (severity) {
    return String(severity).replace('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase())
  }
  if (typeof riskScore === 'number') {
    if (riskScore >= 0.65) return 'Critical'
    if (riskScore >= 0.4) return 'High'
    if (riskScore >= 0.2) return 'Medium'
  }
  return 'Low'
}

function metersBetween(latA, lonA, latB, lonB) {
  const dLat = (latB - latA) * 111_000
  const dLon = (lonB - lonA) * 111_000
  return Math.sqrt(dLat * dLat + dLon * dLon)
}

function pathLengthMeters(path) {
  if (!Array.isArray(path) || path.length < 2) return 0
  let total = 0
  for (let idx = 1; idx < path.length; idx += 1) {
    const prev = path[idx - 1]
    const next = path[idx]
    if (!Array.isArray(prev) || !Array.isArray(next) || prev.length !== 2 || next.length !== 2) continue
    total += metersBetween(prev[0], prev[1], next[0], next[1])
  }
  return total
}

function pathMidpointLatLon(path) {
  if (!Array.isArray(path) || !path.length) return null
  const point = path[Math.floor(path.length / 2)]
  if (!Array.isArray(point) || point.length !== 2) return null
  return { lat: Number(point[0]), lon: Number(point[1]) }
}

function polygonCentroid(polygon) {
  if (!Array.isArray(polygon) || !polygon.length) return null
  const points = polygon.filter((point) => Array.isArray(point) && point.length === 2)
  if (!points.length) return null
  const lat = points.reduce((sum, point) => sum + Number(point[0]), 0) / points.length
  const lon = points.reduce((sum, point) => sum + Number(point[1]), 0) / points.length
  return { lat, lon }
}

function parseRiskFactors(label) {
  if (typeof label !== 'string') return []
  const match = label.match(/\(([^)]+)\)/)
  if (!match?.[1]) return []
  return match[1]
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

function extractHouseholdsEstimate(label) {
  if (typeof label !== 'string') return null
  const match = label.match(/~(\d+)\s*HH/i)
  return match ? Number(match[1]) : null
}

function dedupeList(values) {
  return Array.from(new Set(values.filter(Boolean)))
}

function createStylizedGroundTexture() {
  const canvas = document.createElement('canvas')
  canvas.width = 768
  canvas.height = 768
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return new THREE.CanvasTexture(canvas)
  }

  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height)
  gradient.addColorStop(0, '#244f63')
  gradient.addColorStop(1, '#1a3341')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  for (let i = 0; i < 2200; i += 1) {
    const x = Math.random() * canvas.width
    const y = Math.random() * canvas.height
    const alpha = 0.03 + Math.random() * 0.05
    const size = 0.6 + Math.random() * 1.6
    ctx.fillStyle = `rgba(203,236,248,${alpha})`
    ctx.fillRect(x, y, size, size)
  }

  ctx.lineWidth = 1.8
  for (let i = 0; i < 65; i += 1) {
    const y = (i / 64) * canvas.height
    const alpha = 0.03 + (i % 5 === 0 ? 0.05 : 0.0)
    ctx.strokeStyle = `rgba(170,210,230,${alpha})`
    ctx.beginPath()
    ctx.moveTo(0, y + Math.sin(i * 0.4) * 5)
    ctx.bezierCurveTo(
      canvas.width * 0.25,
      y + Math.cos(i * 0.6) * 14,
      canvas.width * 0.75,
      y + Math.sin(i * 0.3) * 12,
      canvas.width,
      y + Math.cos(i * 0.2) * 4
    )
    ctx.stroke()
  }

  const texture = new THREE.CanvasTexture(canvas)
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.RepeatWrapping
  texture.repeat.set(2.8, 2.2)
  texture.needsUpdate = true
  return texture
}

function seededRandom(seed) {
  let state = seed >>> 0
  return () => {
    state += 0x6d2b79f5
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function toCanvas(lat, lon, bounds, width, height) {
  const x = ((lon - bounds.lonMin) / (bounds.lonMax - bounds.lonMin + 1e-8)) * width
  const y = (1 - (lat - bounds.latMin) / (bounds.latMax - bounds.latMin + 1e-8)) * height
  return [x, y]
}

function drawPathSet(ctx, paths, bounds, width, height, style) {
  const list = Array.isArray(paths) ? paths : []
  list.forEach((path) => {
    if (!Array.isArray(path) || path.length < 2) return
    ctx.beginPath()
    path.forEach((point, idx) => {
      if (!Array.isArray(point) || point.length !== 2) return
      const [x, y] = toCanvas(point[0], point[1], bounds, width, height)
      if (idx === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.lineWidth = style.width
    ctx.strokeStyle = style.color
    ctx.globalAlpha = style.alpha ?? 1
    ctx.stroke()
    ctx.globalAlpha = 1
  })
}

function createSimulatedSatelliteTexture(data, bounds) {
  const canvas = document.createElement('canvas')
  const size = 1024
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return new THREE.CanvasTexture(canvas)
  }

  const seed =
    ((data?.ward?.id ?? 1) * 1103515245 +
      (data?.layers?.houses?.length ?? 0) * 12345 +
      (data?.layers?.roads?.length ?? 0) * 2654435761) >>>
    0
  const rand = seededRandom(seed)

  const gradient = ctx.createLinearGradient(0, 0, 0, size)
  gradient.addColorStop(0, '#4a5f39')
  gradient.addColorStop(0.5, '#5f6e3b')
  gradient.addColorStop(1, '#3c4c2f')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, size, size)

  for (let i = 0; i < 24000; i += 1) {
    const x = rand() * size
    const y = rand() * size
    const w = 0.5 + rand() * 2.2
    const alpha = 0.02 + rand() * 0.08
    const shade = 70 + Math.floor(rand() * 90)
    ctx.fillStyle = `rgba(${shade},${95 + Math.floor(rand() * 55)},${50 + Math.floor(rand() * 35)},${alpha})`
    ctx.fillRect(x, y, w, w)
  }

  for (let i = 0; i < 220; i += 1) {
    const x = rand() * size
    const y = rand() * size
    const radius = 9 + rand() * 34
    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    ctx.fillStyle = `rgba(${90 + Math.floor(rand() * 40)},${105 + Math.floor(rand() * 50)},${65 + Math.floor(rand() * 30)},${0.03 + rand() * 0.06})`
    ctx.fill()
  }

  drawPathSet(ctx, data?.layers?.waterbodies ?? [], bounds, size, size, {
    color: 'rgba(35,79,123,0.82)',
    width: 6.5,
    alpha: 0.95,
  })
  drawPathSet(ctx, data?.layers?.rivers ?? [], bounds, size, size, {
    color: 'rgba(42,106,170,0.88)',
    width: 4.8,
    alpha: 0.95,
  })
  drawPathSet(ctx, data?.layers?.drains ?? [], bounds, size, size, {
    color: 'rgba(72,139,176,0.44)',
    width: 1.5,
    alpha: 0.65,
  })
  drawPathSet(ctx, data?.layers?.roads ?? [], bounds, size, size, {
    color: 'rgba(170,162,149,0.72)',
    width: 1.8,
    alpha: 0.86,
  })

  ;(data?.layers?.informal_zones ?? []).forEach((zone) => {
    const polygon = zone?.polygon
    if (!Array.isArray(polygon) || polygon.length < 3) return
    ctx.beginPath()
    polygon.forEach((point, idx) => {
      if (!Array.isArray(point) || point.length !== 2) return
      const [x, y] = toCanvas(point[0], point[1], bounds, size, size)
      if (idx === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.closePath()
    ctx.fillStyle = 'rgba(155,116,72,0.33)'
    ctx.fill()
  })

  ;(data?.layers?.parks ?? []).forEach((asset) => {
    const [x, y] = toCanvas(asset.lat, asset.lon, bounds, size, size)
    const radius = 2 + Math.min(18, (asset.size ?? 0.1) * 14)
    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(67,139,74,0.72)'
    ctx.fill()
  })

  ;(data?.layers?.playgrounds ?? []).forEach((asset) => {
    const [x, y] = toCanvas(asset.lat, asset.lon, bounds, size, size)
    const radius = 1.2 + Math.min(9, (asset.size ?? 0.08) * 9)
    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(121,166,86,0.7)'
    ctx.fill()
  })

  ;(data?.layers?.houses ?? []).slice(0, 3200).forEach((house) => {
    const [x, y] = toCanvas(house.lat, house.lon, bounds, size, size)
    const buildingSize = 0.9 + Math.min(5.5, (house.height ?? 3) * 0.22)
    ctx.fillStyle = 'rgba(179,167,150,0.66)'
    ctx.fillRect(x - buildingSize * 0.5, y - buildingSize * 0.5, buildingSize, buildingSize)
  })

  ;(data?.layers?.flood_zones ?? []).forEach((zone) => {
    const [x, y] = toCanvas(zone.lat, zone.lon, bounds, size, size)
    const risk = Math.max(0.05, Math.min(1, zone.risk_score ?? 0.2))
    const radius = 8 + risk * 32
    const floodGlow = ctx.createRadialGradient(x, y, 2, x, y, radius)
    floodGlow.addColorStop(0, `rgba(84,164,231,${0.15 + risk * 0.22})`)
    floodGlow.addColorStop(1, 'rgba(84,164,231,0)')
    ctx.fillStyle = floodGlow
    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    ctx.fill()
  })

  for (let i = 0; i < 8; i += 1) {
    const x = rand() * size
    const y = rand() * size
    const rx = 60 + rand() * 150
    const ry = 28 + rand() * 80
    ctx.beginPath()
    ctx.ellipse(x, y, rx, ry, rand() * Math.PI, 0, Math.PI * 2)
    ctx.fillStyle = `rgba(255,255,255,${0.016 + rand() * 0.025})`
    ctx.fill()
  }

  const texture = new THREE.CanvasTexture(canvas)
  texture.wrapS = THREE.ClampToEdgeWrapping
  texture.wrapT = THREE.ClampToEdgeWrapping
  texture.needsUpdate = true
  return texture
}

function createGroundTexture(data, bounds, groundMode) {
  if (groundMode === 'satellite-sim') {
    return createSimulatedSatelliteTexture(data, bounds)
  }
  return createStylizedGroundTexture()
}

function createSkyDome() {
  const geometry = new THREE.SphereGeometry(530, 32, 18)
  const material = new THREE.ShaderMaterial({
    uniforms: {
      topColor: { value: new THREE.Color(0x0e3d62) },
      bottomColor: { value: new THREE.Color(0x9cd1f0) },
      offset: { value: 40 },
      exponent: { value: 0.55 },
    },
    vertexShader: `
      varying vec3 vWorldPosition;
      void main() {
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPosition.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 topColor;
      uniform vec3 bottomColor;
      uniform float offset;
      uniform float exponent;
      varying vec3 vWorldPosition;
      void main() {
        float h = normalize(vWorldPosition + vec3(0.0, offset, 0.0)).y;
        float t = pow(max(h, 0.0), exponent);
        gl_FragColor = vec4(mix(bottomColor, topColor, t), 1.0);
      }
    `,
    side: THREE.BackSide,
    depthWrite: false,
  })
  return new THREE.Mesh(geometry, material)
}

function TwinScene3D({ data, groundMode = 'satellite-sim' }) {
  const stageRef = useRef(null)
  const hostRef = useRef(null)
  const cameraRef = useRef(null)
  const controlsRef = useRef(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [selectedInspector, setSelectedInspector] = useState(null)

  useEffect(() => {
    const onFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === stageRef.current)
      window.dispatchEvent(new Event('resize'))
    }

    document.addEventListener('fullscreenchange', onFullscreenChange)
    return () => {
      document.removeEventListener('fullscreenchange', onFullscreenChange)
    }
  }, [])

  const adjustZoom = (zoomFactor) => {
    const camera = cameraRef.current
    const controls = controlsRef.current
    if (!camera || !controls) return

    const offset = camera.position.clone().sub(controls.target)
    const currentDistance = offset.length()
    const nextDistance = THREE.MathUtils.clamp(
      currentDistance * zoomFactor,
      controls.minDistance,
      controls.maxDistance
    )
    offset.setLength(nextDistance)
    camera.position.copy(controls.target.clone().add(offset))
    camera.updateProjectionMatrix()
    controls.update()
  }

  const toggleFullscreen = async () => {
    const stage = stageRef.current
    if (!stage) return

    try {
      if (document.fullscreenElement === stage) {
        await document.exitFullscreen?.()
        return
      }
      await stage.requestFullscreen?.()
    } catch {
      // Ignore rejected fullscreen requests and keep the scene usable.
    }
  }

  useEffect(() => {
    const host = hostRef.current
    if (!host) return undefined

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(host.clientWidth, host.clientHeight)
    renderer.setClearColor(0x000000, 0)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.08
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    host.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.fog = new THREE.Fog(0x0c2739, 110, 340)
    scene.add(createSkyDome())

    const starsGeometry = new THREE.BufferGeometry()
    const starCount = 380
    const starPositions = new Float32Array(starCount * 3)
    for (let i = 0; i < starCount; i += 1) {
      const radius = 260 + Math.random() * 170
      const theta = Math.random() * Math.PI * 2
      const phi = Math.random() * Math.PI * 0.42
      starPositions[i * 3 + 0] = Math.sin(theta) * Math.sin(phi) * radius
      starPositions[i * 3 + 1] = Math.cos(phi) * radius + 64
      starPositions[i * 3 + 2] = Math.cos(theta) * Math.sin(phi) * radius
    }
    starsGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3))
    const starsMaterial = new THREE.PointsMaterial({
      color: 0xcce9ff,
      size: 1.6,
      transparent: true,
      opacity: 0.45,
      depthWrite: false,
    })
    const stars = new THREE.Points(starsGeometry, starsMaterial)
    scene.add(stars)

    const camera = new THREE.PerspectiveCamera(
      46,
      host.clientWidth / host.clientHeight,
      0.1,
      1000
    )
    camera.position.set(0, 124, 172)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.06
    controls.target.set(0, 3.5, 0)
    controls.maxPolarAngle = Math.PI / 2.02
    controls.minDistance = 64
    controls.maxDistance = 320
    cameraRef.current = camera
    controlsRef.current = controls

    const hemiLight = new THREE.HemisphereLight(0x9ed8ff, 0x2a4b60, 0.64)
    scene.add(hemiLight)
    scene.add(new THREE.AmbientLight(0x84c5eb, 0.24))

    const keyLight = new THREE.DirectionalLight(0xfff9ef, 1.05)
    keyLight.position.set(108, 158, 74)
    keyLight.castShadow = true
    keyLight.shadow.mapSize.set(2048, 2048)
    keyLight.shadow.camera.near = 18
    keyLight.shadow.camera.far = 390
    keyLight.shadow.camera.left = -145
    keyLight.shadow.camera.right = 145
    keyLight.shadow.camera.top = 145
    keyLight.shadow.camera.bottom = -145
    keyLight.shadow.normalBias = 0.02
    scene.add(keyLight)

    const fill = new THREE.DirectionalLight(0x76add0, 0.38)
    fill.position.set(-120, 90, -115)
    scene.add(fill)

    const bounds = collectBounds(data)
    const clickable = []
    const pulsingTargets = []
    const emissivePulses = []
    const spinningTargets = []
    const informalCentroids = (data?.layers?.informal_zones ?? [])
      .map((zone) => {
        const centroid = polygonCentroid(zone?.polygon ?? [])
        if (!centroid) return null
        return {
          lat: centroid.lat,
          lon: centroid.lon,
          households: zone?.households_est ?? 0,
          densityScore: zone?.density_score ?? 0,
        }
      })
      .filter(Boolean)

    const groundTexture = createGroundTexture(data, bounds, groundMode)
    const groundGeometry = new THREE.PlaneGeometry(230, 178, 72, 56)
    const groundVertices = groundGeometry.attributes.position
    for (let i = 0; i < groundVertices.count; i += 1) {
      const x = groundVertices.getX(i)
      const y = groundVertices.getY(i)
      const relief =
        Math.sin(x * 0.052) * 0.85 +
        Math.cos(y * 0.045) * 0.68 +
        Math.sin((x + y) * 0.023) * 0.44
      groundVertices.setZ(i, relief)
    }
    groundGeometry.computeVertexNormals()
    const groundMaterial = new THREE.MeshStandardMaterial({
      map: groundTexture,
      color: groundMode === 'satellite-sim' ? 0xffffff : 0x1e4257,
      metalness: 0.03,
      roughness: 0.9,
    })
    const ground = new THREE.Mesh(groundGeometry, groundMaterial)
    ground.rotation.x = -Math.PI / 2
    ground.position.y = -0.25
    ground.receiveShadow = true
    scene.add(ground)

    const summarizeLocalContext = (lat, lon) => {
      const blockedNearby = (data?.layers?.blocked_drains ?? []).filter((item) =>
        metersBetween(lat, lon, item.lat, item.lon) <= 180
      )
      const nearbyFloodZones = (data?.layers?.flood_zones ?? []).filter((zone) => {
        const distance = metersBetween(lat, lon, zone.lat, zone.lon)
        return distance <= Math.max(zone.radius ?? 0, 90)
      })
      const informalNearby = informalCentroids
        .filter((zone) => metersBetween(lat, lon, zone.lat, zone.lon) <= 260)
        .sort((left, right) => right.households - left.households)[0]

      return {
        blockedNearbyCount: blockedNearby.length,
        maxBlockedRisk: blockedNearby.length
          ? Math.max(...blockedNearby.map((item) => Number(item.risk_score ?? 0)))
          : 0,
        maxFloodRisk: nearbyFloodZones.length
          ? Math.max(...nearbyFloodZones.map((zone) => Number(zone.risk_score ?? 0)))
          : 0,
        informalNearbyHouseholds: informalNearby?.households ?? 0,
      }
    }

    const buildInspector = ({
      title,
      category,
      problem,
      severity,
      riskScore = null,
      label = '',
      lat = null,
      lon = null,
      stats = [],
      riskFactors = [],
    }) => {
      const hasLocation = Number.isFinite(lat) && Number.isFinite(lon)
      const context = hasLocation ? summarizeLocalContext(lat, lon) : null
      const derivedFactors = [...parseRiskFactors(label), ...riskFactors]
      if (context?.maxFloodRisk >= 0.2) {
        derivedFactors.push(`near flood hotspot (${context.maxFloodRisk.toFixed(2)})`)
      }
      if (context?.blockedNearbyCount) {
        derivedFactors.push(`${context.blockedNearbyCount} blocked drain signals nearby`)
      }
      if (context?.informalNearbyHouseholds) {
        derivedFactors.push(
          `near informal-settlement cluster (~${context.informalNearbyHouseholds.toLocaleString()} HH)`
        )
      }

      const detailStats = [...stats]
      if (typeof riskScore === 'number') {
        detailStats.unshift({ label: 'Risk score', value: riskScore.toFixed(2) })
      }
      detailStats.unshift({ label: 'Severity', value: severityLabel(severity, riskScore) })
      if (hasLocation) {
        detailStats.push({ label: 'Location', value: `${lat.toFixed(5)}, ${lon.toFixed(5)}` })
      }
      if (context?.maxFloodRisk >= 0.2) {
        detailStats.push({ label: 'Nearby flood risk', value: context.maxFloodRisk.toFixed(2) })
      }
      if (context?.maxBlockedRisk >= 0.2) {
        detailStats.push({
          label: 'Nearby blockage risk',
          value: context.maxBlockedRisk.toFixed(2),
        })
      }

      return {
        title,
        category,
        problem,
        severity: severityLabel(severity, riskScore),
        riskFactors: dedupeList(derivedFactors).slice(0, 5),
        stats: detailStats,
        note: typeof label === 'string' ? label : '',
      }
    }

    const addLineLayer = ({
      paths,
      color,
      y,
      labelPrefix,
      opacity = 0.86,
      dashed = false,
      maxPaths = Infinity,
      inspectorForPath = null,
      inspectable = true,
    }) => {
      ;(paths ?? []).slice(0, maxPaths).forEach((path, idx) => {
        const points = pathToWorldPoints(path, bounds, y)
        if (points.length < 2) return
        const geometry = new THREE.BufferGeometry().setFromPoints(points)
        const material = dashed
          ? new THREE.LineDashedMaterial({
              color,
              transparent: true,
              opacity,
              dashSize: 2.2,
              gapSize: 1.5,
            })
          : new THREE.LineBasicMaterial({
              color,
              transparent: true,
              opacity,
            })
        const line = new THREE.Line(geometry, material)
        if (dashed) {
          line.computeLineDistances()
        }
        const inspector = inspectorForPath?.(path, idx) ?? null
        line.userData = {
          type: labelPrefix,
          label: `${labelPrefix} segment #${idx + 1}`,
          inspector,
        }
        scene.add(line)
        if (inspectable && inspector) {
          clickable.push(line)
        }
      })
    }

    const addTubeLayer = ({
      paths,
      color,
      y,
      labelPrefix,
      radius,
      maxPaths = 120,
      opacity = 0.9,
      emissive = 0x000000,
      animated = false,
      inspectorForPath = null,
    }) => {
      ;(paths ?? []).slice(0, maxPaths).forEach((path, idx) => {
        const points = pathToWorldPoints(path, bounds, y)
        if (points.length < 2) return
        const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.15)
        const tubularSegments = Math.min(56, Math.max(12, points.length * 3))
        const geometry = new THREE.TubeGeometry(curve, tubularSegments, radius, 9, false)
        const material = new THREE.MeshStandardMaterial({
          color,
          transparent: true,
          opacity,
          metalness: 0.08,
          roughness: 0.48,
          emissive,
          emissiveIntensity: animated ? 0.34 : 0.1,
        })
        const mesh = new THREE.Mesh(geometry, material)
        mesh.receiveShadow = true
        const inspector = inspectorForPath?.(path, idx) ?? null
        mesh.userData = {
          type: labelPrefix,
          label: `${labelPrefix} segment #${idx + 1}`,
          inspector,
        }
        scene.add(mesh)
        if (inspector) {
          clickable.push(mesh)
        }
        if (animated) {
          emissivePulses.push({
            material,
            base: 0.28,
            amplitude: 0.14,
            speed: 1.5 + (idx % 8) * 0.2,
          })
        }
      })
    }

    const addTreeCluster = (x, z, clusterRadius, count, seed) => {
      for (let i = 0; i < count; i += 1) {
        const angle = (Math.PI * 2 * i) / Math.max(count, 1) + seed * 0.41
        const jitter = (Math.sin(seed * 31.7 + i * 17.1) + 1) * 0.5
        const dist = clusterRadius * (0.22 + jitter * 0.68)
        const px = x + Math.cos(angle) * dist
        const pz = z + Math.sin(angle) * dist

        const trunk = new THREE.Mesh(
          new THREE.CylinderGeometry(0.08, 0.11, 0.75, 6),
          new THREE.MeshStandardMaterial({ color: 0x67411f, roughness: 0.9 })
        )
        trunk.position.set(px, 0.64, pz)
        trunk.castShadow = true
        scene.add(trunk)

        const crown = new THREE.Mesh(
          new THREE.ConeGeometry(0.42 + jitter * 0.24, 1.25 + jitter * 0.4, 7),
          new THREE.MeshStandardMaterial({
            color: jitter > 0.55 ? 0x5fb963 : 0x4ea85a,
            roughness: 0.8,
          })
        )
        crown.position.set(px, 1.52 + jitter * 0.18, pz)
        crown.castShadow = true
        scene.add(crown)
      }
    }

    const addCircleAssets = ({ assets, color, y, type, sizeBase, inspectorForAsset = null }) => {
      ;(assets ?? []).forEach((asset, idx) => {
        const point = toWorld(asset.lat, asset.lon, bounds)
        const radius = Math.max(sizeBase, Math.min(sizeBase * 6, (asset.size ?? 0.1) * 3.2))
        const geometry = new THREE.CylinderGeometry(radius, radius, 0.82, 18)
        const material = new THREE.MeshStandardMaterial({
          color,
          metalness: 0.16,
          roughness: 0.54,
          transparent: true,
          opacity: 0.9,
        })
        const mesh = new THREE.Mesh(geometry, material)
        mesh.position.set(point.x, y, point.z)
        mesh.castShadow = true
        mesh.receiveShadow = true
        const inspector = inspectorForAsset?.(asset, idx) ?? null
        mesh.userData = {
          type,
          label: `${type} asset #${idx + 1}`,
          inspector,
        }
        scene.add(mesh)
        if (inspector) {
          clickable.push(mesh)
        }

        if (type === 'Park' && idx < 28) {
          addTreeCluster(point.x, point.z, radius * 0.95, 5 + (idx % 4), idx + 1)
        }

        if (type === 'Playground') {
          const stripe = new THREE.Mesh(
            new THREE.TorusGeometry(radius * 0.72, 0.045, 8, 24),
            new THREE.MeshBasicMaterial({
              color: 0xffffff,
              transparent: true,
              opacity: 0.42,
            })
          )
          stripe.rotation.x = Math.PI / 2
          stripe.position.set(point.x, y + 0.45, point.z)
          scene.add(stripe)
        }
      })
    }

    const boundary = data?.boundary ?? []
    if (boundary.length >= 3) {
      const shape = new THREE.Shape()
      boundary.forEach(([lat, lon], idx) => {
        const point = toWorld(lat, lon, bounds)
        if (idx === 0) {
          shape.moveTo(point.x, point.z)
        } else {
          shape.lineTo(point.x, point.z)
        }
      })

      const geometry = new THREE.ShapeGeometry(shape)
      geometry.rotateX(-Math.PI / 2)
      const material = new THREE.MeshStandardMaterial({
        color: 0x1f4e67,
        transparent: true,
        opacity: 0.5,
        metalness: 0.06,
        roughness: 0.84,
        side: THREE.DoubleSide,
      })
      const polygon = new THREE.Mesh(geometry, material)
      polygon.position.y = 0.14
      polygon.receiveShadow = true
      const boundaryInspector = buildInspector({
        title: data?.ward?.name ?? 'Ward Boundary',
        category: 'Administrative boundary',
        problem: 'Ward extent used to aggregate assets, risks, and policy actions.',
        stats: [
          { label: 'Population', value: (data?.ward?.population ?? 0).toLocaleString() },
          { label: 'Area', value: `${Number(data?.ward?.area_km2 ?? 0).toFixed(2)} km2` },
          { label: 'SDG-11 score', value: Number(data?.ward?.sdg11_score ?? 0).toFixed(2) },
        ],
      })
      polygon.userData = { type: 'Ward Boundary', label: data?.ward?.name ?? 'Ward' }
      polygon.userData.inspector = boundaryInspector
      scene.add(polygon)
      clickable.push(polygon)

      addLineLayer({
        paths: [boundary],
        color: 0xc6efff,
        y: 1.02,
        labelPrefix: 'Ward Edge',
        opacity: 0.75,
        dashed: true,
        maxPaths: 1,
        inspectable: false,
      })
    }

    ;(data?.layers?.informal_zones ?? []).forEach((zone, idx) => {
      const polygon = zone.polygon ?? []
      if (!polygon.length) return
      const shape = new THREE.Shape()
      polygon.forEach(([lat, lon], ptIdx) => {
        const point = toWorld(lat, lon, bounds)
        if (ptIdx === 0) {
          shape.moveTo(point.x, point.z)
        } else {
          shape.lineTo(point.x, point.z)
        }
      })
      const geometry = new THREE.ShapeGeometry(shape)
      geometry.rotateX(-Math.PI / 2)
      const material = new THREE.MeshStandardMaterial({
        color: 0xf3a24b,
        transparent: true,
        opacity: 0.33,
        metalness: 0.05,
        roughness: 0.8,
        emissive: 0x7a451f,
        emissiveIntensity: 0.12,
        side: THREE.DoubleSide,
      })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.y = 0.37 + idx * 0.002
      const centroid = polygonCentroid(polygon)
      const inspector = buildInspector({
        title: `Informal Area Cluster ${idx + 1}`,
        category: 'Settlement vulnerability',
        problem: 'Potential informal settlement cluster with service-access constraints.',
        severity: (zone.density_score ?? 0) >= 0.7 ? 'high' : 'medium',
        riskScore: Number(zone.density_score ?? 0),
        lat: centroid?.lat ?? null,
        lon: centroid?.lon ?? null,
        stats: [
          { label: 'Density score', value: `${Math.round((zone.density_score ?? 0) * 100)}%` },
          { label: 'Estimated households', value: (zone.households_est ?? 0).toLocaleString() },
        ],
        riskFactors: ['dense settlement pattern', 'service access likely constrained'],
      })
      mesh.userData = {
        type: 'Informal Area',
        label: `Informal density ${Math.round((zone.density_score ?? 0) * 100)}% | HH ${zone.households_est ?? 0}`,
        inspector,
      }
      scene.add(mesh)
      clickable.push(mesh)
    })

    ;(data?.layers?.flood_zones ?? []).forEach((zone, idx) => {
      const point = toWorld(zone.lat, zone.lon, bounds)
      const radius = toWorldRadiusMeters(zone.radius ?? 100, bounds)
      const bandColor = severityColor(
        (zone.risk_score ?? 0) > 0.65
          ? 'critical'
          : (zone.risk_score ?? 0) > 0.4
            ? 'high'
            : (zone.risk_score ?? 0) > 0.2
              ? 'medium'
              : 'low'
      )

      const fill = new THREE.Mesh(
        new THREE.CircleGeometry(radius * 0.56, 30),
        new THREE.MeshBasicMaterial({
          color: bandColor,
          transparent: true,
          opacity: 0.11,
          side: THREE.DoubleSide,
        })
      )
      fill.rotation.x = -Math.PI / 2
      fill.position.set(point.x, 0.25 + idx * 0.0012, point.z)
      const estimatedHouseholds = extractHouseholdsEstimate(zone.label)
      const inspector = buildInspector({
        title: `Flood Zone ${idx + 1}`,
        category: 'Flood exposure',
        problem: 'Localized flood-prone area identified by the flood-risk model.',
        riskScore: Number(zone.risk_score ?? 0),
        lat: zone.lat,
        lon: zone.lon,
        label: zone.label,
        stats: [
          { label: 'Impact radius', value: `${Math.round(zone.radius ?? 0)} m` },
          ...(estimatedHouseholds
            ? [{ label: 'Estimated affected households', value: estimatedHouseholds.toLocaleString() }]
            : []),
        ],
      })
      fill.userData = {
        type: 'Flood Exposure Zone',
        label: `${zone.label} | risk ${(zone.risk_score ?? 0).toFixed(2)}`,
        inspector,
      }
      scene.add(fill)
      clickable.push(fill)

      const ring = new THREE.Mesh(
        new THREE.RingGeometry(radius * 0.62, radius, 30),
        new THREE.MeshBasicMaterial({
          color: bandColor,
          transparent: true,
          opacity: 0.34,
          side: THREE.DoubleSide,
        })
      )
      ring.rotation.x = -Math.PI / 2
      ring.position.set(point.x, 0.28 + idx * 0.0014, point.z)
      scene.add(ring)
      spinningTargets.push(ring)

    })

    addTubeLayer({
      paths: data?.layers?.roads ?? [],
      color: 0x7f858e,
      y: 0.72,
      labelPrefix: 'Road',
      radius: 0.24,
      maxPaths: 190,
      opacity: 0.93,
      inspectorForPath: (path, idx) => {
        const midpoint = pathMidpointLatLon(path)
        return buildInspector({
          title: `Road Segment ${idx + 1}`,
          category: 'Transport corridor',
          problem: 'Mobility corridor that may be affected by nearby drainage or flood stress.',
          lat: midpoint?.lat ?? null,
          lon: midpoint?.lon ?? null,
          stats: [{ label: 'Segment length', value: `${Math.round(pathLengthMeters(path))} m` }],
          riskFactors: ['critical access route for neighborhood movement'],
        })
      },
    })
    addLineLayer({
      paths: (data?.layers?.roads ?? []).slice(0, 170),
      color: 0xf5ebcb,
      y: 0.84,
      labelPrefix: 'Road Center',
      opacity: 0.58,
      dashed: true,
      inspectable: false,
    })

    addTubeLayer({
      paths: data?.layers?.drains ?? [],
      color: 0x52c9e6,
      y: 0.53,
      labelPrefix: 'Drain',
      radius: 0.14,
      maxPaths: 130,
      opacity: 0.82,
      emissive: 0x1b6376,
      animated: true,
      inspectorForPath: (path, idx) => {
        const midpoint = pathMidpointLatLon(path)
        return buildInspector({
          title: `Drain Line ${idx + 1}`,
          category: 'Drain infrastructure',
          problem: 'Drainage line monitored for congestion and blockage pressure.',
          lat: midpoint?.lat ?? null,
          lon: midpoint?.lon ?? null,
          stats: [{ label: 'Line length', value: `${Math.round(pathLengthMeters(path))} m` }],
          riskFactors: ['surface runoff channel', 'sensitive to blockage and maintenance delays'],
        })
      },
    })

    addTubeLayer({
      paths: data?.layers?.rivers ?? [],
      color: 0x2f78de,
      y: 0.44,
      labelPrefix: 'River',
      radius: 0.3,
      maxPaths: 90,
      opacity: 0.86,
      emissive: 0x17408a,
      animated: true,
      inspectorForPath: (path, idx) => {
        const midpoint = pathMidpointLatLon(path)
        return buildInspector({
          title: `River / Canal Segment ${idx + 1}`,
          category: 'Blue infrastructure',
          problem: 'Water corridor influencing local flood exposure and drainage performance.',
          lat: midpoint?.lat ?? null,
          lon: midpoint?.lon ?? null,
          stats: [{ label: 'Segment length', value: `${Math.round(pathLengthMeters(path))} m` }],
          riskFactors: ['water overflow exposure', 'adjacent drainage dependence'],
        })
      },
    })

    addTubeLayer({
      paths: data?.layers?.waterbodies ?? [],
      color: 0x2e82df,
      y: 0.36,
      labelPrefix: 'Waterbody',
      radius: 0.38,
      maxPaths: 95,
      opacity: 0.62,
      emissive: 0x184f84,
      animated: true,
      inspectorForPath: (path, idx) => {
        const midpoint = pathMidpointLatLon(path)
        return buildInspector({
          title: `Waterbody Edge ${idx + 1}`,
          category: 'Waterbody',
          problem: 'Standing-water edge that can intensify flood or drainage stress nearby.',
          lat: midpoint?.lat ?? null,
          lon: midpoint?.lon ?? null,
          stats: [{ label: 'Edge length', value: `${Math.round(pathLengthMeters(path))} m` }],
          riskFactors: ['adjacent runoff accumulation', 'water proximity'],
        })
      },
    })

    ;(data?.layers?.blocked_drain_network ?? []).forEach((segment, idx) => {
      const points = pathToWorldPoints(segment?.path ?? [], bounds, 0.87)
      if (points.length < 2) return
      const risk = Number(segment.risk_score ?? 0)
      const color =
        risk >= 0.65 ? 0xee3f3f : risk >= 0.4 ? 0xf86f31 : risk >= 0.2 ? 0xffa743 : 0xffd36a
      const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.2)
      const geometry = new THREE.TubeGeometry(curve, Math.max(8, points.length * 4), 0.08 + risk * 0.08, 8, false)
      const material = new THREE.MeshStandardMaterial({
        color,
        transparent: true,
        opacity: 0.95,
        metalness: 0.1,
        roughness: 0.35,
        emissive: color,
        emissiveIntensity: 0.42,
      })
      const mesh = new THREE.Mesh(geometry, material)
      const midpoint = pathMidpointLatLon(segment?.path ?? [])
      const inspector = buildInspector({
        title: `Blocked Drain Segment ${idx + 1}`,
        category: 'Drain blockage',
        problem: 'Predicted blockage on the drainage network requiring clearing or maintenance.',
        severity: segment.severity,
        riskScore: Number(segment.risk_score ?? 0),
        lat: midpoint?.lat ?? null,
        lon: midpoint?.lon ?? null,
        label: segment.label,
        stats: [{ label: 'Segment length', value: `${Math.round(pathLengthMeters(segment?.path ?? []))} m` }],
      })
      mesh.userData = {
        type: 'Blocked Drain Network',
        label: `${segment.label ?? `Segment ${idx + 1}`} | risk ${risk.toFixed(2)}`,
        inspector,
      }
      scene.add(mesh)
      clickable.push(mesh)
      emissivePulses.push({
        material,
        base: 0.36 + risk * 0.2,
        amplitude: 0.16,
        speed: 2.1 + risk,
      })

    })

    ;(data?.layers?.houses ?? []).forEach((house, idx) => {
      const point = toWorld(house.lat, house.lon, bounds)
      const height = Math.max(1.8, Math.min(20, house.height ?? 4))
      const width = Math.max(0.72, Math.min(2.7, (house.footprint ?? 0.001) * 130))
      const facadeColor = new THREE.Color().setHSL(
        0.075 + (idx % 9) * 0.012,
        0.33,
        0.41 + (idx % 5) * 0.03
      )
      const building = new THREE.Mesh(
        new THREE.BoxGeometry(width, height, width),
        new THREE.MeshStandardMaterial({
          color: facadeColor,
          metalness: 0.03,
          roughness: 0.84,
        })
      )
      building.position.set(point.x, height * 0.5 + 0.2, point.z)
      building.castShadow = true
      building.receiveShadow = true
      const inspector = buildInspector({
        title: `House ${idx + 1}`,
        category: 'Residential asset',
        problem: 'Residential asset exposed to surrounding service and climate risks.',
        lat: house.lat,
        lon: house.lon,
        stats: [
          { label: 'Estimated height', value: `${height.toFixed(1)} m` },
          { label: 'Footprint index', value: Number(house.footprint ?? 0).toFixed(3) },
        ],
        riskFactors: ['residential exposure', 'depends on nearby drainage performance'],
      })
      building.userData = {
        type: 'House',
        label: `House #${idx + 1} | est. height ${height.toFixed(1)}m`,
        inspector,
      }
      scene.add(building)
      clickable.push(building)

      const roof = new THREE.Mesh(
        new THREE.ConeGeometry(width * 0.78, Math.max(0.8, height * 0.24), 4),
        new THREE.MeshStandardMaterial({
          color: 0x6b4a35,
          roughness: 0.74,
        })
      )
      roof.rotation.y = Math.PI / 4
      roof.position.set(point.x, height + 0.52, point.z)
      roof.castShadow = true
      scene.add(roof)
    })

    addCircleAssets({
      assets: data?.layers?.playgrounds ?? [],
      color: 0x48d28f,
      y: 0.58,
      type: 'Playground',
      sizeBase: 1.0,
      inspectorForAsset: (asset, idx) =>
        buildInspector({
          title: `Playground ${idx + 1}`,
          category: 'Public amenity',
          problem: 'Community facility that can be affected by flood, blockage, or settlement stress nearby.',
          lat: asset.lat,
          lon: asset.lon,
          stats: [{ label: 'Amenity size', value: Number(asset.size ?? 0).toFixed(2) }],
          riskFactors: ['public-space usability risk'],
        }),
    })
    addCircleAssets({
      assets: data?.layers?.parks ?? [],
      color: 0x7ee17a,
      y: 0.5,
      type: 'Park',
      sizeBase: 1.26,
      inspectorForAsset: (asset, idx) =>
        buildInspector({
          title: `Park ${idx + 1}`,
          category: 'Green asset',
          problem: 'Green space that buffers heat and runoff but may lose function under neighborhood hazards.',
          lat: asset.lat,
          lon: asset.lon,
          stats: [{ label: 'Amenity size', value: Number(asset.size ?? 0).toFixed(2) }],
          riskFactors: ['green-access asset', 'sensitive to flood and maintenance stress'],
        }),
    })

    ;(data?.layers?.blocked_drains ?? []).forEach((spot, idx) => {
      const point = toWorld(spot.lat, spot.lon, bounds)
      const color = severityColor(spot.severity)
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.98, 14, 12),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.3,
        })
      )
      mesh.position.set(point.x, 2.25, point.z)
      const inspector = buildInspector({
        title: `Blocked Drain Marker ${idx + 1}`,
        category: 'Drain blockage signal',
        problem: 'High-risk blockage marker on the modeled drainage network.',
        severity: spot.severity,
        riskScore: Number(spot.risk_score ?? 0),
        lat: spot.lat,
        lon: spot.lon,
        label: spot.label,
      })
      mesh.userData = {
        type: 'Blocked Drain',
        label: `${spot.label} | risk ${(spot.risk_score ?? 0).toFixed(2)}`,
        inspector,
      }
      scene.add(mesh)
      clickable.push(mesh)
      pulsingTargets.push({
        mesh,
        baseScale: 1,
        amplitude: 0.13,
        speed: 2.3 + (idx % 6) * 0.12,
      })

    })

    ;(data?.hotspots ?? []).forEach((spot, idx) => {
      const point = toWorld(spot.lat, spot.lon, bounds)
      const color = severityColor(spot.severity)
      const mesh = new THREE.Mesh(
        new THREE.ConeGeometry(0.92, 2.55, 12),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.22,
          roughness: 0.45,
        })
      )
      mesh.position.set(point.x, 1.78, point.z)
      mesh.castShadow = true
      const inspector = buildInspector({
        title: spot.issue,
        category: 'Problem hotspot',
        problem: `${spot.issue} identified from the ward evidence pipeline.`,
        severity: spot.severity,
        lat: spot.lat,
        lon: spot.lon,
        stats: [{ label: 'Evidence source', value: spot.source }],
        riskFactors: [spot.source.replace('-', ' ')],
      })
      mesh.userData = {
        type: 'Problem Hotspot',
        label: `${spot.issue} (${spot.severity})`,
        inspector,
      }
      scene.add(mesh)
      clickable.push(mesh)
      pulsingTargets.push({
        mesh,
        baseScale: 1,
        amplitude: 0.11,
        speed: 2.8 + (idx % 5) * 0.2,
      })

    })

    const actionMarkers = data?.actions_taken ?? []
    actionMarkers.slice(0, 8).forEach((action, idx) => {
      const path = data?.layers?.roads?.[idx] ?? data?.layers?.drains?.[idx] ?? []
      if (!path.length) return
      const [lat, lon] = path[Math.floor(path.length / 2)]
      const point = toWorld(lat, lon, bounds)
      const color =
        action.status === 'completed'
          ? 0x7af0ac
          : action.status === 'in_progress'
            ? 0xffca5c
            : 0xaac4d7

      const pole = new THREE.Mesh(
        new THREE.CylinderGeometry(0.08, 0.08, 2.6, 8),
        new THREE.MeshStandardMaterial({ color: 0xd9e2ea, roughness: 0.52 })
      )
      pole.position.set(point.x, 1.48, point.z)
      scene.add(pole)

      const flag = new THREE.Mesh(
        new THREE.BoxGeometry(1.4, 0.72, 0.08),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.15,
        })
      )
      flag.position.set(point.x + 0.72, 2.2, point.z)
      const inspector = buildInspector({
        title: action.title,
        category: 'Policy action',
        problem: 'Selected intervention responding to local service and exposure constraints.',
        lat,
        lon,
        stats: [
          { label: 'Status', value: action.status.replace('_', ' ') },
          { label: 'Progress', value: `${action.progress_pct}%` },
          { label: 'Agency', value: action.agency },
          { label: 'Cost', value: `${Number(action.estimated_cost_lakh ?? 0).toFixed(2)} lakh` },
          {
            label: 'Expected beneficiaries',
            value: (action.expected_beneficiaries ?? 0).toLocaleString(),
          },
        ],
        riskFactors: [action.category],
      })
      flag.userData = {
        type: 'Action',
        label: `${action.title} | ${action.status} ${action.progress_pct}%`,
        inspector,
      }
      scene.add(flag)
      clickable.push(flag)
    })

    const raycaster = new THREE.Raycaster()
    raycaster.params.Line.threshold = 2.4
    const pointer = new THREE.Vector2()

    const onSceneClick = (event) => {
      const rect = renderer.domElement.getBoundingClientRect()
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(pointer, camera)
      const intersections = raycaster.intersectObjects(clickable, true)
      if (!intersections.length) {
        setSelectedInspector(null)
        return
      }
      const inspector = intersections[0].object?.userData?.inspector
      setSelectedInspector(inspector ?? null)
    }
    renderer.domElement.addEventListener('click', onSceneClick)

    const onResize = () => {
      const width = host.clientWidth
      const height = host.clientHeight
      renderer.setSize(width, height)
      camera.aspect = width / Math.max(height, 1)
      camera.updateProjectionMatrix()
    }
    window.addEventListener('resize', onResize)

    const clock = new THREE.Clock()
    let rafId = 0
    const animate = () => {
      const elapsed = clock.getElapsedTime()
      stars.rotation.y = elapsed * 0.01

      pulsingTargets.forEach((entry, idx) => {
        const scale = entry.baseScale * (1 + Math.sin(elapsed * entry.speed + idx) * entry.amplitude)
        entry.mesh.scale.set(scale, scale, scale)
      })

      emissivePulses.forEach((entry, idx) => {
        entry.material.emissiveIntensity =
          entry.base + Math.sin(elapsed * entry.speed + idx) * entry.amplitude
      })

      spinningTargets.forEach((mesh, idx) => {
        mesh.rotation.z = elapsed * (0.28 + idx * 0.012)
      })

      controls.update()
      renderer.render(scene, camera)
      rafId = window.requestAnimationFrame(animate)
    }
    animate()

    return () => {
      window.cancelAnimationFrame(rafId)
      window.removeEventListener('resize', onResize)
      renderer.domElement.removeEventListener('click', onSceneClick)
      controls.dispose()
      if (cameraRef.current === camera) {
        cameraRef.current = null
      }
      if (controlsRef.current === controls) {
        controlsRef.current = null
      }
      scene.traverse((object) => {
        if (object.geometry) {
          object.geometry.dispose()
        }
        if (Array.isArray(object.material)) {
          object.material.forEach((material) => {
            if (material?.map) {
              material.map.dispose()
            }
            material.dispose()
          })
        } else if (object.material) {
          if (object.material.map) {
            object.material.map.dispose()
          }
          object.material.dispose()
        }
      })
      renderer.dispose()
      if (renderer.domElement.parentNode === host) {
        host.removeChild(renderer.domElement)
      }
    }
  }, [data, groundMode])

  return (
    <div
      className={`scene-canvas-wrap${isFullscreen ? ' is-fullscreen' : ''}`}
      ref={stageRef}
    >
      <div className="scene-controls">
        <button
          type="button"
          className="scene-control"
          onClick={() => adjustZoom(0.82)}
          aria-label="Zoom in"
          title="Zoom in"
        >
          +
        </button>
        <button
          type="button"
          className="scene-control"
          onClick={() => adjustZoom(1.18)}
          aria-label="Zoom out"
          title="Zoom out"
        >
          -
        </button>
        <button
          type="button"
          className="scene-control scene-control-wide"
          onClick={toggleFullscreen}
          aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
          title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
        >
          {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
        </button>
      </div>
      {selectedInspector ? (
        <div className="scene-inspector">
          <div className="scene-inspector-head">
            <div>
              <span className="scene-inspector-kicker">{selectedInspector.category}</span>
              <strong>{selectedInspector.title}</strong>
            </div>
            <button
              type="button"
              className="scene-inspector-close"
              onClick={() => setSelectedInspector(null)}
              aria-label="Close inspector"
            >
              x
            </button>
          </div>
          <p className="scene-inspector-problem">{selectedInspector.problem}</p>
          {selectedInspector.riskFactors?.length ? (
            <div className="scene-inspector-section">
              <span>Risk factors</span>
              <ul>
                {selectedInspector.riskFactors.map((factor) => (
                  <li key={factor}>{factor}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {selectedInspector.stats?.length ? (
            <div className="scene-inspector-stats">
              {selectedInspector.stats.map((item) => (
                <div key={`${item.label}-${item.value}`}>
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>
          ) : null}
          {selectedInspector.note ? (
            <p className="scene-inspector-note">{selectedInspector.note}</p>
          ) : null}
        </div>
      ) : (
        <div className="scene-hint">Click a feature to inspect problem, risk factors, and stats.</div>
      )}
      <div className="scene-canvas" ref={hostRef} />
      {!data && <div className="scene-placeholder">Loading digital twin scene...</div>}
    </div>
  )
}

export default TwinScene3D
