-- Adds noise to input data by randomly removing few values
-- sets output as original image
-- train denosing autoencoder

require 'optim'
require 'nn'

-- parse command line arguments
if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Train denoising autoencoder given input')
   cmd:text()
   cmd:option('-inputdata','nil','Input data file in ASCII')
   cmd:text()
   opt = cmd:parse(arg or {})
end

batch_size = 100
dofile('load_data.lua')
local data_reader = load_data(opt.inputdata,batch_size)

ninput = 256*256
nhidden = (30)

--ipdata = ipdata:float()
--opdata  = ipdata:copy()
torch.setdefaulttensortype('torch.DoubleTensor')


model = nn.Sequential()
model:add(nn.Linear(ninput,nhidden))
model:add(nn.Sigmoid())
model:add(nn.Linear(nhidden,ninput))


-- Loss function
criterion = nn.MSECriterion()
criterion.sizeAverage = true


parameters,gradParameters = model:getParameters()


optimState = {
   learningRate = 0.1
}



function train()

   epoch = epoch or 1

   local time=sys.clock()

   local batch_err = 0
   local tot_err = 0
   local num_batches = 0

   data_reader:reinit()
   
   -- do one epoch
   for t = 1, data_reader:input_size(), batch_size do
      local inputs = data_reader:next_batch()
      --display progress
      xlua.progress(math.min(data_reader:current_ptr()+batch_size-1,data_reader:input_size()), data_reader:input_size())
      
      
      local targets = inputs
      -- create minibatch
      -- local inputs = ipdata
      -- local targets = ipdata
      -- ipdata = nil
      -- --local inputs = {}
      --local targets = {}
      --for i = t, math.min(t+batch_size-1, ipdata:size(1)) do
	 --local input = ipdata[i]
	 --local target = ipdata[i]
	 --table.insert(inputs, input)
	 --table.insert(targets, target)
      --end

      -- create closure to evalueate f(X) and df(X)/dX
      local feval = function(x)
	 --get new parameters
	 if x ~= parameters then
	    parameters:copy(x)
	 end

	 --reset gradients
	 gradParameters:zero()

	 batch_err = 0

	 -- f is the average of all criterions
	 local f = 0

	 -- evaluate function for complete minibatch
	 for i = 1,#inputs do
	    --estimate f
       local corrupted_input = nn.Dropout():forward(torch.Tensor(inputs[i]))
	    local output = model:forward(corrupted_input)
	    local err = criterion:forward(output, targets[i])
	    f = f + err
	    batch_err = batch_err + err

	    -- estimate df/dW
	    local df_do = criterion:backward(output, targets[i])
	    model:backward(corrupted_input, df_do)

	 end

	 batch_err = batch_err/#inputs
	 --print('Avg error on minibatch: '..batch_err)
	 tot_err = tot_err + batch_err

	 -- normalize gradients and f(X)
	 gradParameters:div(#inputs)
	 f = f/#inputs

	 -- return f and df/dX
	 return f,gradParameters
      end

      -- optimize on current minibatch
      optim.adagrad(feval, parameters, optimState)
      num_batches = num_batches + 1
      
   end

   tot_err = tot_err/num_batches
   print('Training Error= '.. tot_err)

   -- time taken
   time = sys.clock()-time
   print('Time taken= '.. time)

   epoch = epoch + 1
   
end

for j = 1, 10 do
   train()  
end
data_reader:reinit()
ipdata=image.load('../data/train/000012.jpg')
ipdata=nn.Reshape(256*256):forward(image.rgb2y(ipdata))
torch.save('net.bin', model)
weight = torch.Tensor((model:forward(ipdata)))
image.save('test.jpg',weight:reshape(256,256))
image.save('input.jpg',ipdata:reshape(256,256))
