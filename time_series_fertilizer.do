import excel "C:\Users\Iván Trejo\OneDrive\DataScience\Proyectos\time-series-fertilizer-price-predictor\data\predicción_precios_fertilizantes.xlsx", sheet("Sheet1") firstrow clear

tsset date
gen date_daily = dofc(date)     // Convierte %tc a fecha diaria (%td)
format date_daily %td           // Aplica formato diario
gen date_monthly = mofd(date_daily)   // Convierte a número de mes
format date_monthly %tm               // Aplica formato mensual
tsset date_monthly


drop if missing(fob_urea_kg) | missing(crude_oil_brent_usd_per_bbl) | missing(natural_gas_europe_usd_per_mmbtu) | missing(inflación_mensual) | missing(ipc_diesel) | missing(precipitación_media_mm) | missing(temp_media_Celsius) | missing(fob_otros_prod_agrícolas_exporta) | missing(fosfatodiamónico18460) | missing(muriatodepotasio0060) | missing(urea4600) 

regress urea4600 fob_urea_kg crude_oil_brent_usd_per_bbl natural_gas_europe_usd_per_mmbtu inflación_mensual ipc_diesel precipitación_media_mm temp_media_Celsius fob_otros_prod_agrícolas_exporta, vce(robust)


estat sbsingle