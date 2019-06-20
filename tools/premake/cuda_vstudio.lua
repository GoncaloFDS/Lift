--
-- cuda_vstudio.lua
-- CUDA integration for vstudio.
-- Copyright (c) 2012-2015 Manu Evans and the Premake project
--

	local p = premake
	local m = p.modules.cuda

	m.elements = {}

	local vstudio = p.vstudio
	local vc2010 = vstudio.vc2010

	m.element = vc2010.element


---- C/C++ projects ----

	function m.cudaPropertyGroup(prj)
		p.push('<PropertyGroup>')
		m.element("CUDAPropsPath", "Condition=\"'$(CUDAPropsPath)'==''\"", "$(VCTargetsPath)\\BuildCustomizations")
		p.pop('</PropertyGroup>')
	end
	p.override(vc2010.elements, "project", function(oldfn, prj)
		local sections = oldfn(prj)
		table.insertafter(sections, vc2010.project, m.cudaPropertyGroup)
		return sections
	end)

	function m.cudaToolkitPath(prj)
		p.w("<CudaToolkitCustomDir />")
	end
	p.override(vc2010.elements, "globals", function(oldfn, prj)
		local globals = oldfn(prj)
		table.insertafter(globals, m.cudaToolkitPath)
		return globals
	end)


	function m.cudaProps(prj)
		p.w("<Import Project=\"$(CUDAPropsPath)\\CUDA 8.0.props\" />")
	end
	p.override(vc2010.elements, "importExtensionSettings", function(oldfn, prj)
		local importExtensionSettings = oldfn(prj)
		table.insert(importExtensionSettings, m.cudaProps)
		return importExtensionSettings
	end)


	function m.cudaRuntime(cfg)
		if cfg.cudaruntime then
--			m.element("JSONFile", nil, cfg.cudaruntime)
		end
	end

	m.elements.cudaCompile = function(cfg)
		return {
			m.cudaRuntime
		}
	end

	function m.cudaCompile(cfg)
		p.push('<CudaCompile>')
		p.callArray(m.elements.cudaCompile, cfg)
		p.pop('</CudaCompile>')
	end
	p.override(vc2010.elements, "itemDefinitionGroup", function(oldfn, cfg)
		local cuda = oldfn(cfg)
		table.insertafter(cuda, vc2010.clCompile, m.cudaCompile)
		return cuda
	end)

	function m.cudaTargets(prj)
		p.w("<Import Project=\"$(CUDAPropsPath)\\CUDA 8.0.targets\" />")
	end
	p.override(vc2010.elements, "importExtensionTargets", function(oldfn, prj)
		local targets = oldfn(prj)
		table.insert(targets, m.cudaTargets)
		return targets
	end)


---
-- CudaCompile group
---
	vc2010.categories.CudaCompile = {
		name       = "CudaCompile",
		extensions = { ".cu" },
		priority   = 2,

		emitFiles = function(prj, group)
			local fileCfgFunc = function(fcfg, condition)
				if fcfg then
					return {
						vc2010.excludedFromBuild,
						-- TODO: D per-file options
--						m.objectFileName,
--						m.clCompilePreprocessorDefinitions,
--						m.clCompileUndefinePreprocessorDefinitions,
--						m.optimization,
--						m.forceIncludes,
--						m.precompiledHeader,
--						m.enableEnhancedInstructionSet,
--						m.additionalCompileOptions,
--						m.disableSpecificWarnings,
--						m.treatSpecificWarningsAsErrors
					}
				else
					return {
						vc2010.excludedFromBuild
					}
				end
			end

			vc2010.emitFiles(prj, group, "CudaCompile", {vc2010.generatedFile}, fileCfgFunc)
		end,

		emitFilter = function(prj, group)
			vc2010.filterGroup(prj, group, "CudaCompile")
		end
	}
