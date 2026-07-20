# Data dictionary
## features_5yr.csv  (780 samples x 35 features + cls)
cls: 1=broadleaf, 2=conifer, 3=bamboo.
Optical (Sentinel-2): NDVI/NDMI/NBR/reNDVI each _mean,_amp,_pha (harmonic mean/amplitude/phase).
SAR (Sentinel-1): VV_mean,VH_mean,RATIO_mean,RATIO_std,RVI_mean,VH_amp,VH_pha,VV_wetdry,VH_wetdry,RATIO_wetdry.
Climate (ERA5-Land): temp_mean,dewp_mean,dewdep,soilm_mean/wet/dry/wetdry,precip_mm/wet/dry.
Terrain (SRTM): elev,slope,aspect_cos.
## features_train4yr.csv / features_test1yr_2025-2026.csv  independent spatio-temporal validation windows.
## train_points.csv  lon,lat (WGS84),cls; row-aligned with features_5yr.csv.
## cloud_monthly_stats_5yr.csv  monthly S2 cloud stats: mean_cloud,n_clear20,n_scenes,month,year.
## misclassified_points.csv  89 misclassified points with coords, attributes, entropy, official type, confusion features.
## feature_importance_5fold.csv  5-fold feature importance (gini, perm) + group.
