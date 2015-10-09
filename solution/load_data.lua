-- Read the image files store in dirname directory.
-- each image is added to the table data
-- data table is then returned.

require 'paths'
require 'image'
require 'torch'
require 'nn'
local load_data  = torch.class('load_data')

function load_data:__init(dirname,batch_size)
	--count number of images
	self.batch_size = batch_size
	self.dirname = dirname
	self.file_list = {}
	self._start_batch = 1
	self.ninput = 256*256

	local i = 0

   	for line in paths.files(dirname) do
      	if line:match(".jpg") then
      		i = i + 1
      		table.insert(self.file_list,paths.concat(dirname,line))
      	end
   	end
   	self._nimage = i
end

function load_data:next_batch()
	local nrow = math.min(self._nimage,self._start_batch+self.batch_size)
	local data = {}--torch.Tensor(nrow,ninput)
	local i = self._start_batch
	while i <= nrow do
		f=self.file_list[i]
		value = nn.Reshape(ninput):forward(image.rgb2y(image.load(f)))
		
		table.insert(data,value)
		--data[i] = value
		i = i + 1
	end
	self._start_batch = nrow
	return data
	-- body
end
   
function load_data:reinit()
	self._start_batch = 1 
end

function  load_data:input_size()
	return self._nimage
end

function load_data:current_ptr()
	return self._start_batch
end