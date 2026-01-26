SELECT 
	gvkey,
	datadate AS date,
	prccd as Close,
	cshtrd as SharesOutstanding
FROM comp.secd
WHERE 
	datadate BETWEEN %s AND %s  -- Filter for dates in specific range
    AND tpci = '0'              -- Common shares only (filter out ETFs and similar)
    AND prccd IS NOT NULL
    AND cshtrd IS NOT NULL
    AND cshtrd > 0;