SELECT 
    DISTINCT a.gvkey,
    a.conm as CompanyName,
    a.sic as SICCode,
    a.busdesc as BusinessDescription,
    a.spcindcd as SnPIndustrySector,
    a.spcseccd as SnPEcpnSector,
    b.tic as Ticker
FROM comp.company as a 
INNER JOIN comp.secm as b 
    ON a.gvkey = b.gvkey
WHERE 
    b.tpci = '0'  -- Common shares only
    AND b.tic IS NOT NULL
    AND a.sic IS NOT NULL;