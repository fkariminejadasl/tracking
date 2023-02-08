## Access Snellius GPUs

### Create an account

Create ticket or email servicedesk@surf.nl
https://servicedesk.surf.nl 

https://portal.cua.surf.nl : first copied public key in here (only done once)

### Usage
Use the Snellius (similar for e.g. sshfs/scp):
ssh -X username@snellius.surf.nl

#### Run a job and check
squeue -u username -o "%.18i %.9P %.18j %.8u %.2t %.10M %.6D %.10R %.20S %.4p"
