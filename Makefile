.PHONY: fetch fetch-dry up down psql clean-pdfs

fetch:
	python scripts/fetch_stdlib.py

fetch-dry:
	python scripts/fetch_stdlib.py --dry-run

up:
	docker compose up -d

down:
	docker compose down

psql:
	docker compose exec db psql -U omg -d omg_rag

clean-pdfs:
	rm -f data/pdfs/*.pdf data/pdfs/download_meta.json
