env=FourRooms-misc

python3 foundIT_VLM.py -env $env 
python3 utils/helpers.py --env $env --num 1 --samp 3 --iter 1

python3 foundIT_VLM.py -env $env 
python3 utils/helpers.py --env $env --num 2 --samp 3 --iter 1

python3 foundIT_VLM.py -env $env 
python3 utils/helpers.py --env $env --num 3 --samp 3 --iter 1

python3 foundIT_VLM.py -env $env -c
python3 utils/helpers.py --env $env --num c1 --samp 3 --iter 1

python3 foundIT_VLM.py -env $env -c
python3 utils/helpers.py --env $env --num c2 --samp 3 --iter 1

python3 foundIT_VLM.py -env $env -c
python3 utils/helpers.py --env $env --num c3 --samp 3 --iter 2

python3 foundIT_VLM.py -env $env 
python3 utils/helpers.py --env $env --num 1 --samp 3 --iter 2

python3 foundIT_VLM.py -env $env 
python3 utils/helpers.py --env $env --num 2 --samp 3 --iter 2

python3 foundIT_VLM.py -env $env 
python3 utils/helpers.py --env $env --num 3 --samp 3 --iter 2

python3 foundIT_VLM.py -env $env -c
python3 utils/helpers.py --env $env --num c1 --samp 3 --iter 2

python3 foundIT_VLM.py -env $env -c
python3 utils/helpers.py --env $env --num c2 --samp 3 --iter 2

python3 foundIT_VLM.py -env $env -c
python3 utils/helpers.py --env $env --num c3 --samp 5 --iter 3

python3 foundIT_VLM.py -env $env 
python3 utils/helpers.py --env $env --num 1 --samp 5 --iter 3

python3 foundIT_VLM.py -env $env 
python3 utils/helpers.py --env $env --num 2 --samp 5 --iter 3

python3 foundIT_VLM.py -env $env 
python3 utils/helpers.py --env $env --num 3 --samp 5 --iter 3

python3 foundIT_VLM.py -env $env -c
python3 utils/helpers.py --env $env --num c1 --samp 5 --iter 3

python3 foundIT_VLM.py -env $env -c
python3 utils/helpers.py --env $env --num c2 --samp 5 --iter 3

python3 foundIT_VLM.py -env $env -c
python3 utils/helpers.py --env $env --num c3 --samp 5 --iter 3