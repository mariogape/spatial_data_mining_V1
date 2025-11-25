# ---- 0. Paquetes ----
# Instala una vez si hace falta:
# install.packages("easyclimate")
# install.packages("terra")

library(easyclimate)
library(terra)

# ---- 1. Paths ----
# Usa barras normales o dobles barras invertidas en Windows
aoi_dir  <- "C:/Darwin geospatial/25.01 OpenPAS/AOI Shapefiles/4326"
out_dir  <- "C:/Darwin geospatial/25.01 OpenPAS/Climate_Tmean"

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 2. Listar AOIs (shapefiles) ----
# Si usas GeoPackage, cambia el patrón a "*.gpkg"
aoi_files <- list.files(aoi_dir, pattern = "\\.shp$", full.names = TRUE)

if (length(aoi_files) == 0) {
  stop("No he encontrado ningún .shp en la carpeta de AOIs.")
}

# Barra de progreso para seguir el avance por AOI
total_aoi <- length(aoi_files)
pb <- utils::txtProgressBar(min = 0, max = total_aoi, style = 3)

# ---- 3. Definir periodo climático ----
# Opción tipo climatología: 1991–2020
# period_dates <- 2019:2020    # cambia esto si quieres otro periodo

# También podrías usar fechas:
period_dates <- "2020-12-01:2020-12-31"

# ---- 4. Bucle sobre AOIs ----
for (i in seq_along(aoi_files)) {
  shp <- aoi_files[i]
  message("Procesando AOI ", i, "/", total_aoi, ": ", shp)

  # 4.1 Leer AOI como SpatVector (terra)
  aoi <- vect(shp)  # AOI en lon/lat WGS84 (idealmente)

  # 4.2 Descargar datos diarios de Tmin y Tmax como raster
  #    output = "raster" devuelve una lista de SpatRaster (uno por variable) :contentReference[oaicite:1]{index=1}
  clim_rasters <- get_daily_climate(
    coords        = aoi,
    climatic_var  = c("Tmin", "Tmax"),
    period        = period_dates,
    output        = "raster"
  )

  # clim_rasters es una lista: $Tmin y $Tmax, cada uno multilayer (un layer por día)
  tmin_r <- clim_rasters$Tmin
  tmax_r <- clim_rasters$Tmax

  # 4.3 Temperatura media diaria (Tmean) = (Tmin + Tmax)/2
  # Terra hace la suma por layers (mismo nº de capas, misma extensión).
  tmean_daily <- (tmin_r + tmax_r) / 2

  # 4.4 Media en todo el periodo (climatología de Tmean)
  # mean() sobre un SpatRaster multilayer promedia en el eje temporal
  tmean_clim <- mean(tmean_daily, na.rm = TRUE)

  # 4.5 Enmascarar al AOI (por si acaso quieres todo fuera del polígono como NA)
  tmean_masked <- mask(tmean_clim, aoi)

  # 4.6 Guardar a disco (GeoTIFF)
  base_name <- tools::file_path_sans_ext(basename(shp))
  out_file  <- file.path(out_dir,
                         paste0("Tmean_", min(period_dates), "-", max(period_dates),
                                "_", base_name, ".tif"))

  writeRaster(tmean_masked, out_file, overwrite = TRUE)
  message("  -> escrito: ", out_file)
  utils::setTxtProgressBar(pb, i)
}

close(pb)
message("Listo. Tienes un raster de Tmedia anual por AOI en: ", out_dir)
