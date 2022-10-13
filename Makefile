build_api:
	docker build -t doc.smartparking.kz/lp_recognizer_api:0.0.1.2.22.2 api/

build_bot:
	docker build -t doc.smartparking.kz/lp_recognizer_bot:0.0.1.2.22.1 telegram/

run:
	docker-compose up -d

stop:
	docker-compose down

restart:
	make stop
	make run

build_all:
	make build_api
	#make build_bot

build_and_run:
	make build_all
	make run
